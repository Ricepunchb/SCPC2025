import os
import re
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
from itertools import chain
import wandb

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler 
from transformers import T5ForConditionalGeneration, T5Tokenizer, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, load_from_disk, ClassLabel

from ..customdatasets import *


class WeightedSFTTrainer(SFTTrainer):
    def __init__(self, train_weights, eval_weights, *args, **kwargs):
        super().__init__(*args, **kwargs) # 부모 클래스의 __init__ 호출
        self.train_weights = train_weights # WeightedSFTTrainer에만 필요한 weights 추가
        self.eval_weights = eval_weights
        
    def get_train_dataloader(self) -> DataLoader:
        # 데이터셋 준비
        train_dataset = self.train_dataset
        
        # 1단계에서 계산한 가중치를 사용해 샘플러 생성
        sampler = WeightedRandomSampler(
            weights=self.train_weights,
            num_samples=len(train_dataset),
            replacement=True
        )

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def get_eval_dataloader(self, eval_dataset = None) -> DataLoader:
        sampler = WeightedRandomSampler(
            weights=self.eval_weights,
            num_samples=len(eval_dataset),
            replacement=True
        )

        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory
        )
        
        

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

    # "Answer: A" 또는 "Answer: B" 처럼 Answer: 뒤에 직접 오는 경우
    match_answer_prefix = re.search(r'Answer:\s*([A-D])(?!\S)', text) # A, B, C, D 뒤에 공백이나 문장이 끝나는 경우
    if match_answer_prefix:
        return match_answer_prefix.group(1)
    
    # "(C)." 처럼 괄호 안에 오는 경우
    match_answer_prefix = re.search(r'\s*\((\s*[A-D]\s*)\)\.?', text) # A, B, C, D 뒤에 공백이나 문장이 끝나는 경우
    if match_answer_prefix:
        return match_answer_prefix.group(1)

    # 모델이 단순히 'A' 또는 'B'만 생성했을 때를 대비하는 것입니다.
    # 예: text = "A"
    if text.strip() in ['A', 'B', 'C', 'D']:
        return text.strip()

    return None # 어떤 패턴도 찾지 못한 경우 None 반환


def compute_accuracy(output, tokenizer):
    pred_texts = []
    for batch_preds in output.predictions:
        logits = batch_preds[0]
        token_ids = np.argmax(logits, axis=-1)
        decoded_preds = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        pred_texts.extend(decoded_preds)

    label_texts = []
    for sample in output.label_ids:     
        encoded_labels = np.where(sample == -100, tokenizer.pad_token_id, sample)
        decoded_labels = tokenizer.batch_decode(encoded_labels, skip_special_tokens=True) # batch size의 decoded text list가 됨
        label_texts.extend(decoded_labels)

    assert len(pred_texts) == len(label_texts), "Prediction and label counts must match"
    
    # decode
    predicted_texts = tokenizer.batch_decode(pred_texts, skip_special_tokens=True)
    actual_texts = tokenizer.batch_decode(label_texts, skip_special_tokens=True)

    # accuracy 계산
    correct, total = 0, 0
    for p_text, a_text in zip(predicted_texts, actual_texts):
        p_ans = extract_answer_letter(p_text)
        a_ans = extract_answer_letter(a_text)

        if not a_ans:
            continue
        total += 1
        if p_ans and p_ans == a_ans:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "Test_Accuracy": round(accuracy, 4),
        "Total_preds": total,
        "Correct_preds": correct
    }


def main():
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
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map='auto', _attn_implementation="eager",)
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large", add_eos_token=True)
    data_collator = T5DataCollator(tokenizer)
    
    # # --------------------- [Optional] 모델 로드 for continual training -----------------------
    # final_save_directory = os.path.join("Model/flan-t5/final_best_model")
    # model = T5ForConditionalGeneration.from_pretrained(final_save_directory, torch_dtype=torch.bfloat16, device_map="auto")   # 모델 로드
    # tokenizer = T5Tokenizer.from_pretrained(final_save_directory)

    # --------------------- Training Config -----------------------
    batch_size = 8
    training_args = SFTConfig(
        report_to='wandb',
        run_name = 'run-0722',
        output_dir="Model/flan-t5/1st_phase",
        overwrite_output_dir=True,
        num_train_epochs=3,     # epochs
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=128 // batch_size,
        eval_accumulation_steps=4,
        warmup_ratio=0.05,
        learning_rate=1e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        lr_scheduler_kwargs = {"num_cycles": 2.5},
        logging_steps=10,
        eval_strategy='steps',
        eval_steps=0.1,
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=1,
        metric_for_best_model="Test_Accuracy",
        greater_is_better=True,
        load_best_model_at_end=True,
        optim="adamw_torch",
        remove_unused_columns=False,
        gradient_checkpointing=True,   
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        label_names=["labels"],
        max_grad_norm=1.0,
        eval_do_concat_batches=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True
        )

    # --------------------- DACON 데이터셋 --------------------
    ds = load_dataset("json", data_files="eg/train_aug.jsonl", split='train')
    ds = ds.train_test_split(test_size=0.2)
    val_1 = DACONDataset(ds['test'])
    train_1 = DACONDataset(ds['train'])

    # --------------------- Visual 7w -------------------------------
    # Visual 7w 데이터셋 로드
    ds = load_dataset("json", data_files="datasets/visual7w/visual7w_augmented_with_descriptions.jsonl", split='train')
    train_2, val_2 = preproc_visual7w(ds)
    train_2 = VisualDataset(train_2)
    val_2 = VisualDataset(val_2)
    
    # --------------------- Realworld QA -------------------------------
    # Realworld QA 데이터셋 로드
    ds = load_from_disk("datasets/Realworld/Realworld_aug")
    ds = ds.filter(lambda x: x['answer'] in ['A', 'B', 'C', 'D'])
    ds = ds.class_encode_column("answer").train_test_split(test_size=0.1, stratify_by_column='answer')
    train_3 = RealworldDataset(ds['train'])
    val_3 = RealworldDataset(ds['test'])
    
    # --------------------- A-OKVQA -------------------------------
    train_4 = load_dataset("HuggingFaceM4/A-OKVQA", split="train")
    val_4 = load_dataset("HuggingFaceM4/A-OKVQA", split="validation")
    train_4 = AokvqaDataset(train_4)
    val_4 = AokvqaDataset(val_4)

 # --------------------- VMC Dataset -------------------------------
    ds = load_from_disk("datasets/VMC/VMC_aug")
    class_labels = ClassLabel(names=["A", "B", "C", "D"])
    ds = ds.cast_column("answer", class_labels)
    ds = ds.train_test_split(test_size=0.2, stratify_by_column='answer')
    train_5, val_5 = ds['train'], ds['test']
    train_5 = VMCDataset(train_5)
    val_5 = VMCDataset(val_5)
    
    train_ds = ConcatDataset([train_1, train_2, train_3, train_4, train_5])
    val_ds = ConcatDataset([val_1, val_2, val_3, val_4, val_5])
    print(f"Total number of train samples: {len(train_ds)}")
    print(f"Total number of validation samples: {len(val_ds)}")
    train_weights = [[1.0/len(d)] * len(d) for d in [train_1, train_2, train_3, train_4, train_5] ]
    train_weights = list(chain.from_iterable(train_weights))
    assert len(train_weights) == len(train_ds)
    
    val_weights = [[1.0/len(d)] * len(d) for d in [val_1, val_2, val_3, val_4, val_5] ]
    val_weights = list(chain.from_iterable(val_weights))
    
    del ds, train_1, train_2, train_3, train_4, train_5, val_1, val_2, val_3, val_4, val_5
    torch.cuda.empty_cache()
    gc.collect()
    
    trainer = WeightedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        train_weights=train_weights,
        eval_weights=val_weights,
        compute_metrics=lambda eval_pred: compute_accuracy(eval_pred, tokenizer)
    )
    
    print("✅ 1st Training start.")
    trainer.train()
    print("✅ 1st Training done.")
    
    # 훈련 결과 저장
    final_save_directory = os.path.join(training_args.output_dir, "final_best_model")
    os.makedirs(final_save_directory, exist_ok=True)
    trainer.model.save_pretrained(final_save_directory)     # 모델 저장
    print(f"✅ Model saved to {final_save_directory}")
    tokenizer.save_pretrained(final_save_directory)     # tokenizer 저장
    print(f"✅ Tokenizer saved to {final_save_directory}")
    
    # 베스트 모델 로드
    model = T5ForConditionalGeneration.from_pretrained(final_save_directory, device_map="auto")   # 모델 로드
    tokenizer = T5Tokenizer.from_pretrained(final_save_directory)
      
    #--------------------------------------- Reasoning 학습 --------------------------------
    ds = load_dataset("tau/commonsense_qa")
    train_ds, val_ds = ds['train'], ds['validation']
    train_ds, val_ds = ECQADataset(train_ds), ECQADataset(val_ds)
    
    del ds, trainer, training_args
    torch.cuda.empty_cache()
    gc.collect()
    
    training_args = SFTConfig(
        report_to='wandb',
        run_name = 'run-0722',
        output_dir="Model/flan-t5/2nd_phase",
        overwrite_output_dir=True,
        num_train_epochs=3,     # epochs
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=128 // batch_size,
        eval_accumulation_steps=4,
        warmup_ratio=0.05,
        learning_rate=5e-6,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        lr_scheduler_kwargs = {"num_cycles": 2.5},
        logging_steps=10,
        eval_strategy='steps',
        eval_steps=0.1,
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=1,
        metric_for_best_model="Test_Accuracy",
        greater_is_better=True,
        load_best_model_at_end=True,
        optim="adamw_torch",
        remove_unused_columns=False,
        gradient_checkpointing=True,   
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        label_names=["labels"],
        max_grad_norm=1.0,
        eval_do_concat_batches=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True
        )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_accuracy(eval_pred, tokenizer)
    )
    
    print("✅ 2nd Training start.")
    trainer.train()
    print("✅ 2nd Training done.")
    
    # 훈련 결과 저장
    final_save_directory = os.path.join(training_args.output_dir, "final_best_model")
    os.makedirs(final_save_directory, exist_ok=True)
    trainer.model.save_pretrained(final_save_directory)     # 모델 저장
    print(f"✅ Model saved to {final_save_directory}")
    tokenizer.save_pretrained(final_save_directory)     # tokenizer 저장
    print(f"✅ Tokenizer saved to {final_save_directory}")
    
    # 베스트 모델 로드
    model = T5ForConditionalGeneration.from_pretrained(final_save_directory, device_map="auto")   # 모델 로드
    tokenizer = T5Tokenizer.from_pretrained(final_save_directory)
    
    print("✅ Done.")
    # ----------------------------------------------- 제출용 추론 --------------------------------------------------
    # 메모리 정리
    del train_ds, val_ds, trainer
    torch.cuda.empty_cache() # GPU 캐시 비우기
    gc.collect()             # Python 가비지 컬렉터 실행
    
    # 모델 정보
    total_params = sum(p.numel() for p in model.parameters())
    print(f"모델 정보:")
    print(f"총 파라미터 수: {total_params:,}")


if __name__ == "__main__":
    main()