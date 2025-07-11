import os
import re
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import gc
import wandb

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, Kosmos2ForConditionalGeneration, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import accuracy_score
from datasets import load_dataset, load_from_disk, ClassLabel
from peft import LoraConfig, PeftModel, PeftConfig

from ..customdatasets import *


# 정답 알파벳 추출 함수
def extract_answer_letter(text):
    """
    주어진 텍스트에서 최종 답변 알파벳 (예: '(A)')을 추출합니다.
    모델의 출력 패턴에 따라 정규 표현식 등을 사용하여 더 견고하게 만들 수 있습니다.
    """
    # 괄호와 점이 포함된 패턴: "the final answer is: (A)." 또는 "Answer: (A)."
    match = re.search(r'(?:the final answer is:|Answer:)\s*\((\s*[A-D]\s*)\)\.?', text)
    if match:
        return match.group(1).strip()

    # 괄호만 포함된 패턴: "the final answer is: (A)" 또는 "Answer: (A)"
    match_paren_only = re.search(r'(?:the final answer is:|Answer:)\s*\((\s*[A-D]\s*)\)', text)
    if match_paren_only:
        return match_paren_only.group(1).strip()

    # 2. "Answer: A" 또는 "Answer: B" 처럼 Answer: 뒤에 직접 오는 경우
    match_answer_prefix = re.search(r'Answer:\s*([A-D])(?!\S)', text) # A, B, C, D 뒤에 공백이나 문장이 끝나는 경우
    if match_answer_prefix:
        return match_answer_prefix.group(1)

    # 모델이 단순히 'A' 또는 'B'만 생성했을 때를 대비하는 것입니다.
    # 예: text = "A"
    if text.strip() in ['A', 'B', 'C', 'D']:
        return text.strip()

    return None # 어떤 패턴도 찾지 못한 경우 None 반환


def preprocess_logits_for_metrics(logits, labels):
    if logits is None:
        print(f"Warning: preprocess_logits_for_metrics received None for logits!")
        return None, labels
    if isinstance(logits, tuple) and logits[0] is None:
        print(f"Warning: preprocess_logits_for_metrics received a tuple with None at logits[0]!")
        return None, labels
    
    pred_ids = torch.argmax(logits[0], dim=-1)
    
    return pred_ids, labels


class Compute_Metrics:
    def __init__(self, processor):
        self.tokenizer = processor.tokenizer
        
    def __call__(self, pred):
        '''
        A,B,C,D 중 최종 정답 맞춘 갯수로 정답율
        '''
        if isinstance(pred.predictions, tuple):
            preds = pred.predictions[1] 
        else:
            preds = pred.predictions
        labels = pred.label_ids
        if preds.size == 0 or labels.size == 0:
            print("Warning: preds or labels array is empty. Returning 0.0 accuracy.")
            return {"accuracy": 0.0}

        preds_decoded  = []
        labels_decoded  = []
        
        for i in range(preds.shape[0]):
            valid_pred_ids = preds[i][preds[i] != -100] 
            preds_decoded.append(self.tokenizer.decode(valid_pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))
            
            valid_label_ids = labels[i][labels[i] != -100]
            labels_decoded.append(self.tokenizer.decode(valid_label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))
            
        # print("Decoded Preds: ", preds_decoded[:5])
        # print("Decoded Labels: ", labels_decoded[:5])
        
        pred_answers = []
        label_answers = []
        
        for i in range(len(preds_decoded)):
            # 정답 텍스트에서 'answer' 부분만 추출:
            label_text = labels_decoded[i]
            label_answer = extract_answer_letter(label_text)
            label_answers.append(label_answer)

            # 예측 텍스트에서 'answer' 부분만 추출:
            pred_text = preds_decoded[i]
            pred_answer = extract_answer_letter(pred_text)
            pred_answers.append(pred_answer)
            
        # print("Answer Preds: ", pred_answers[:5])
        # print("Answer Labels: ", label_answers[:5])

        # accuracy 계산
        correct_count = 0
        total_count = 0
        for i in range(len(label_answers)):
            p_ans = pred_answers[i]
            l_ans = label_answers[i]

            if l_ans is not None: # 유효한 정답이 있는 경우에만 계산 (None은 파싱 실패)
                total_count += 1
                # 예측도 유효하고, 정답과 일치하는 경우
                if p_ans is not None and p_ans == l_ans:
                    correct_count += 1
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        print(f"Calculated accuracy: {accuracy} (Correct: {correct_count}, Total Valid Samples: {total_count})")

        return {"accuracy": accuracy}
        

def main():
    # # !! Important HOTFIX !!
    # import transformers.models.kosmos2.modeling_kosmos2 as kosmos2_module

    # original_forward_embedding = kosmos2_module.Kosmos2TextTransformer.forward_embedding

    # def patched_forward_embedding(self, input_ids, inputs_embeds, image_embeds, img_input_mask, past_key_values_length, position_ids):
    #     if inputs_embeds is not None:
    #         inputs_embeds = inputs_embeds.clone()
    #     return original_forward_embedding(
    #         self, input_ids, inputs_embeds, image_embeds,
    #         img_input_mask, past_key_values_length, position_ids
    #     )

    # kosmos2_module.Kosmos2TextTransformer.forward_embedding = patched_forward_embedding

    # WandB 시동
    wandb.init(project="SCPC 2025")

    # 환경 설정
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ Using device:", device)

    # 시드 고정
    def seed_everything(seed=47):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed_everything()

    # --------------------- 모델 선언 -----------------------
    model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224", device_map='auto', torch_dtype=torch.bfloat16, _attn_implementation="sdpa",)
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # 모델 정보
    total_params = sum(p.numel() for p in model.parameters())
    print(f"모델 정보:")
    print(f"총 파라미터 수: {total_params:,}")

    # --------------------- SFTTrainer 설정 -----------------------
    batch_size = 20
    training_args = SFTConfig(
        report_to='wandb',
        run_name = 'run-0711',
        output_dir="Model/kosmos2",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=128 // batch_size,
        eval_accumulation_steps=128 // batch_size,
        # warmup_steps=128,
        learning_rate=1e-4,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=100,
        eval_strategy='epoch',
        save_strategy="best",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        optim="adamw_torch_fused",
        bf16=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        label_names=["labels"]
        )
    early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3, 
    early_stopping_threshold=0.005,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "out_proj",  # attention
            "fc1", "fc2",  # feedforward
            "dense",  # image_to_text_projection dense layer
        ],
        exclude_modules=["vision_model"],
        lora_dropout=0.05,
        bias="none",
    )

    # --------------------- 데이터 collator 선언 -----------------------
    data_collator = DataCollator(processor)
    compute_metrics = Compute_Metrics(processor)

    # --------------------- DACON 데이터셋 TEST 데이터로 --------------------
    test_ds = load_dataset("json", data_files="eg/train_aug.jsonl", split='train')
    test_ds = DACONDataset(test_ds)

    # --------------------- Visual 7w -------------------------------
    # Visual 7w 데이터셋 로드
    visual_ds = load_dataset("json", data_files="datasets/visual7w/visual7w_augmented_with_descriptions.jsonl", split='train')
    train_ds, val_ds = preproc_visual7w(visual_ds)
    train_ds = VisualDataset(train_ds)
    val_ds = VisualDataset(val_ds)
    
    # 메모리 정리
    del visual_ds       
    torch.cuda.empty_cache() # GPU 캐시 비우기
    gc.collect()             # Python 가비지 컬렉터 실행

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[early_stopping_callback]
    )
    
    trainer.train()
    
    preds_obj = trainer.predict(test_ds)
    
    # Eval with Test ds
    preds_obj = trainer.predict(test_ds)
    print("Test accuracy: \n", compute_metrics(preds_obj))
    
    # 모델 수동 저장
    final_save_directory = os.path.join(training_args.output_dir, "final_best_model")
    os.makedirs(final_save_directory, exist_ok=True)
    trainer.model.save_pretrained(final_save_directory)     # 이것은 PEFT 어댑터 (LoRA 가중치)와 PEFT 설정만 저장
    print(f"PEFT adapter saved to {final_save_directory}")
    processor.save_pretrained(final_save_directory)     # processor 저장
    print(f"Processor saved to {final_save_directory}")
    
    # 베스트 모델 로드
    model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224", torch_dtype=torch.bfloat16, device_map="auto" )   # 베이스모델 로드
    model = PeftModel.from_pretrained(model, final_save_directory, torch_dtype=torch.bfloat16, device_map="auto", is_trainable=True )        # LoRA weight 적용된 모델 로드
    processor = AutoProcessor.from_pretrained(final_save_directory)
    data_collator = DataCollator(processor)
    
    # --------------------- Realworld QA -------------------------------
    # Realworld QA 데이터셋 로드
    ds = load_from_disk("datasets/Realworld/Realworld_aug")
    ds = ds.filter(lambda x: x['answer'] in ['A', 'B', 'C', 'D'])
    ds = ds.class_encode_column("answer").train_test_split(test_size=0.2, stratify_by_column='answer')
    train_ds = RealworldDataset(ds['train'])
    val_ds = RealworldDataset(ds['test'])
    
    # 메모리 정리
    del ds
    torch.cuda.empty_cache() # GPU 캐시 비우기
    gc.collect()             # Python 가비지 컬렉터 실행

    training_args.learning_rate = 5e-5      # lr 재조정
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        # peft_config=lora_config,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[early_stopping_callback]
    )
    
    trainer.train()
    
    # Eval with Test ds
    preds_obj = trainer.predict(test_ds)
    print("Test accuracy: \n", compute_metrics(preds_obj))
    
    # 모델 수동 저장
    final_save_directory = os.path.join(training_args.output_dir, "final_best_model")
    os.makedirs(final_save_directory, exist_ok=True)
    trainer.model.save_pretrained(final_save_directory)     # 이것은 PEFT 어댑터 (LoRA 가중치)와 PEFT 설정만 저장
    print(f"PEFT adapter saved to {final_save_directory}")
    processor.save_pretrained(final_save_directory)     # processor 저장
    print(f"Processor saved to {final_save_directory}")
    
    # 베스트 모델 로드
    model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224", torch_dtype=torch.bfloat16, device_map="auto" )   # 베이스모델 로드
    model = PeftModel.from_pretrained(model, final_save_directory, torch_dtype=torch.bfloat16, device_map="auto", is_trainable=True )        # LoRA weight 적용된 모델 로드
    processor = AutoProcessor.from_pretrained(final_save_directory)
    data_collator = DataCollator(processor)
    
    # --------------------- A-OKVQA -------------------------------
    train_ds = load_dataset("HuggingFaceM4/A-OKVQA", split="train")
    val_ds = load_dataset("HuggingFaceM4/A-OKVQA", split="validation")
    train_ds = AokvqaDataset(train_ds)
    val_ds = AokvqaDataset(val_ds)
    
    training_args.learning_rate = 1e-5      # lr 재조정
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        # peft_config=lora_config,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[early_stopping_callback]
    )

    trainer.train()
    
    preds_obj = trainer.predict(test_ds)
    
    # Eval with Test ds
    preds_obj = trainer.predict(test_ds)
    print("Test accuracy: \n", compute_metrics(preds_obj))
    
    # 모델 수동 저장
    final_save_directory = os.path.join(training_args.output_dir, "final_best_model")
    os.makedirs(final_save_directory, exist_ok=True)
    trainer.model.save_pretrained(final_save_directory)     # 이것은 PEFT 어댑터 (LoRA 가중치)와 PEFT 설정만 저장
    print(f"PEFT adapter saved to {final_save_directory}")
    processor.save_pretrained(final_save_directory)     # processor 저장
    print(f"Processor saved to {final_save_directory}")
    
    # 베스트 모델 로드
    model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224", torch_dtype=torch.bfloat16, device_map="auto" )   # 베이스모델 로드
    model = PeftModel.from_pretrained(model, final_save_directory, torch_dtype=torch.bfloat16, device_map="auto", is_trainable=True )        # LoRA weight 적용된 모델 로드
    processor = AutoProcessor.from_pretrained(final_save_directory)
    data_collator = DataCollator(processor)

 # --------------------- VMC Dataset -------------------------------
    ds = load_from_disk("datasets/VMC/VMC_aug")
    
    class_labels = ClassLabel(names=["A", "B", "C", "D"])
    ds = ds.cast_column("answer", class_labels)
    ds = ds.train_test_split(test_size=0.2, stratify_by_column='answer')
    train_ds, val_ds = ds['train'], ds['test']
    train_ds = VMCDataset(train_ds)
    val_ds = VMCDataset(val_ds)
    
    del ds   # 메모리 정리    
    torch.cuda.empty_cache() # GPU 캐시 비우기
    gc.collect()             # Python 가비지 컬렉터 실행
    
    training_args.learning_rate = 5e-6      # lr 재조정
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        # peft_config=lora_config,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[early_stopping_callback]
    )

    trainer.train()
    
    preds_obj = trainer.predict(test_ds) # predict 메서드는 EvalPrediction 객체를 반환
    
    # Eval with Test ds
    preds_obj = trainer.predict(test_ds)
    print("Test accuracy: \n", compute_metrics(preds_obj))
    
    # 모델 수동 저장
    final_save_directory = os.path.join(training_args.output_dir, "final_best_model")
    os.makedirs(final_save_directory, exist_ok=True)
    trainer.model.save_pretrained(final_save_directory)     # 이것은 PEFT 어댑터 (LoRA 가중치)와 PEFT 설정만 저장
    print(f"PEFT adapter saved to {final_save_directory}")
    processor.save_pretrained(final_save_directory)     # processor 저장
    print(f"Processor saved to {final_save_directory}")
    
    # 베스트 모델 로드
    peft_config = PeftConfig.from_pretrained(final_save_directory)     # 훈련후 best checkpoint
    model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224", torch_dtype=torch.bfloat16, device_map="auto" )   # 베이스모델 로드
    model = PeftModel.from_pretrained(model, final_save_directory, torch_dtype=torch.bfloat16, device_map="auto", is_trainable=True )        # LoRA weight 적용된 모델 로드
    processor = AutoProcessor.from_pretrained(final_save_directory)
    data_collator = DataCollator(processor)
    
    # ----------------------------------------------- 제출용 추론 --------------------------------------------------
    # 메모리 정리
    del train_ds, val_ds, test_ds, trainer
    torch.cuda.empty_cache() # GPU 캐시 비우기
    gc.collect()             # Python 가비지 컬렉터 실행
    
    test = pd.read_csv('eg/test.csv')
    results = []
    
    model.eval()
    
    for _, row in tqdm(test.iterrows(), total=len(test)):
        image = Image.open('eg/'+row['img_path']).convert("RGB")
        choices = '\n'.join([f"{c}. {row[c]}" for c in ['A', 'B', 'C', 'D']])

        prompt = f"Question: {row['Question']}. Choose the correct option. \n Options: {choices} \n Answer: "

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        output = model.generate(
            # pixel_values=inputs["pixel_values"],
            # input_ids=inputs["input_ids"],
            # attention_mask=inputs["attention_mask"],
            # image_embeds_position_mask=inputs["image_embeds_position_mask"],
            **inputs,
            image_embeds=None,
            use_cache=True,
            do_sample=False,
            # top_k=5,
            # top_p=0.9,
            # temperature=0.5,
            min_new_tokens=25,      
            max_new_tokens=128,
            num_beams=3
            )
        generated_text = processor.batch_decode(output, skip_special_tokens=True)[0]
        answer, entities = processor.post_process_generation(generated_text)
        results.append(extract_answer_letter(answer))
        tqdm.write(f"Generated: {answer} \n")

    print('✅ Inference Done.')

    # submission용 CSV 만들기
    submission = pd.read_csv('eg/sample_submission.csv')
    submission['answer'] = results
    submission.to_csv('eg/baseline_submit.csv', index=False)
    print("✅ CSV for submission Done.")

if __name__ == "__main__":
    main()