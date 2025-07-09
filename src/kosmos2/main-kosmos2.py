import os
import re
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import wandb

import torch
from torch import nn
from sklearn.metrics import accuracy_score
from transformers import AutoProcessor, AutoModelForVision2Seq, Kosmos2ForConditionalGeneration, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig

from datasets import load_dataset
from peft import LoraConfig, PeftModel, PeftConfig

from ..customdatasets import AokvqaDataset, DACONDataset, VisualDataset, DataCollator, preproc_visual7w


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    
    return {"accuracy": accuracy}


# 정답 알파벳 추출 함수
def extract_answer_letter(text):
    match = re.search(r"\s*([A-Da-d])\s*", text)
    return match.group(1).upper() if match else "?"


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
    training_args = SFTConfig(
        report_to='wandb',
        run_name = 'run-0709-1',
        output_dir="Model/kosmos2",
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=256,
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_steps=25,
        eval_strategy='epoch',
        save_strategy="best",
        optim="adamw_torch_fused",
        bf16=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True}
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

    # --------------------- dacon 데이터셋 TEST 데이터로 --------------------
    test_ds = load_dataset("csv", data_files="eg/train.csv", split='train')
    test_ds = DACONDataset(test_ds, processor)

    # --------------------- Visual 7w -------------------------------
    # Visual 7w 데이터셋 로드
    visual_ds = load_dataset("json", data_files="datasets/visual7w/dataset_v7w_telling.json", split='train')
    train_ds, val_ds = preproc_visual7w(visual_ds)
    train_ds = VisualDataset(train_ds, processor)
    val_ds = VisualDataset(val_ds, processor)

    del visual_ds       # 메모리 정리

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    
    preds = trainer.predict(test_ds)
    logits = preds.predictions
    labels = preds.label_ids
    y_pred = np.argmax(logits, axis=-1) # 예측값 (최대 확률을 가지는 class 선택)
    acc = accuracy_score(labels, y_pred)    # 정확도 출력
    print(f"Validation Accuracy: {acc:.4f}")
    
    best_ckpt = trainer.state.best_model_checkpoint
    print("Best checkpoint saved at:", best_ckpt)
    
    peft_config = PeftConfig.from_pretrained(best_ckpt)     # 훈련후 best checkpoint
    model = Kosmos2ForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto" )
    model = PeftModel.from_pretrained(model, best_ckpt, torch_dtype=torch.bfloat16, device_map="auto" )        # LoRA weight 적용된 모델 로드
    
    # --------------------- A-OKVQA -------------------------------
    train_ds = load_dataset("HuggingFaceM4/A-OKVQA", split="train")
    val_ds = load_dataset("HuggingFaceM4/A-OKVQA", split="validation")
    train_ds = AokvqaDataset(train_ds, processor)
    val_ds = AokvqaDataset(val_ds, processor)
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    
    preds = trainer.predict(test_ds)
    logits = preds.predictions
    labels = preds.label_ids
    y_pred = np.argmax(logits, axis=-1) # 예측값 (최대 확률을 가지는 class 선택)
    acc = accuracy_score(labels, y_pred)    # 정확도 출력
    print(f"Validation Accuracy: {acc:.4f}")
    
    best_ckpt = trainer.state.best_model_checkpoint
    print("Best checkpoint saved at:", best_ckpt)
    
    peft_config = PeftConfig.from_pretrained(best_ckpt)     # 훈련후 best checkpoint
    model = Kosmos2ForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto" )
    model = PeftModel.from_pretrained(model, best_ckpt, torch_dtype=torch.bfloat16, device_map="auto" )        # LoRA weight 적용된 모델 로드

    # ----------------------------------------------- 제출용 추론 --------------------------------------------------
    model = model.merge_and_unload()   # 모델 merge
    model.save_pretrained("Model/merged_model")
    processor.save_pretrained("Model/merged_model")  # 추천
    model = Kosmos2ForConditionalGeneration.from_pretrained("Model/merged_model", torch_dtype=torch.bfloat16, device_map="auto" )
    processor = AutoProcessor.from_pretrained("Model/merged_model")
    
    test = pd.read_csv('eg/test.csv')
    results = []

    for _, row in tqdm(test.iterrows(), total=len(test)):
        image = Image.open('eg/'+row['img_path']).convert("RGB")
        choices = '\n'.join([f"{c}. {row[c]}" for c in ['A', 'B', 'C', 'D']])

        prompt = f"Question: {row['Question']}. Choose the correct option. \n Options: {choices} \n Answer: "

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        output = model.generate(
            **inputs,
            do_sample=True,
            top_k=5,
            top_p=0.9,
            temperature=0.5,      
            max_new_tokens=5,
            )
        decoded = processor.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        answer = extract_answer_letter(decoded)
        results.append(answer)
        tqdm.write(f"{decoded}")
        tqdm.write(f"Answer: {answer} \n")

    print('✅ Inference Done.')

    # submission용 CSV 만들기
    submission = pd.read_csv('eg/sample_submission.csv')
    submission['answer'] = results
    submission.to_csv('eg/baseline_submit.csv', index=False)
    print("✅ CSV for submission Done.")

if __name__ == "__main__":
    main()