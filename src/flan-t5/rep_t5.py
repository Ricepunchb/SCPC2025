import os
import re
import random
import warnings
import numpy as np
import gc

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, load_from_disk, ClassLabel

from ..customdatasets import *

        

# 정답 알파벳 추출 함수
def extract_answer_letter(text):
    """
    주어진 텍스트에서 최종 답변 알파벳 (예: '(A)')을 추출합니다.
    모델의 출력 패턴에 따라 정규 표현식 등을 사용하여 더 견고하게 만들 수 있습니다.
    """
    # 괄호와 점이 포함된 패턴: "the final answer is: (A)." 또는 "Answer: (A)."
    match = re.search(r'(?:The answer is:|Answer:)\s*\((\s*[A-E]\s*)\)\.?', text)
    if match:
        return match.group(1).strip()

    # 괄호만 포함된 패턴: "the final answer is: (A)" 또는 "Answer: (A)"
    match_paren_only = re.search(r'(?:The answer is:|Answer:)\s*\((\s*[A-E]\s*)\)', text)
    if match_paren_only:
        return match_paren_only.group(1).strip()

    # "Answer: A" 또는 "Answer: B" 처럼 Answer: 뒤에 직접 오는 경우
    match_answer_prefix = re.search(r'Answer:\s*([A-E])(?!\S)', text) # A, B, C, D 뒤에 공백이나 문장이 끝나는 경우
    if match_answer_prefix:
        return match_answer_prefix.group(1)
    
    # "(C)." 처럼 괄호 안에 오는 경우
    match_answer_prefix = re.search(r'\s*\((\s*[A-E]\s*)\)\.?', text) # A, B, C, D 뒤에 공백이나 문장이 끝나는 경우
    if match_answer_prefix:
        return match_answer_prefix.group(1)

    # 모델이 단순히 'A' 또는 'B'만 생성했을 때를 대비
    # 예: text = "A"
    if text.strip() in ['A', 'B', 'C', 'D', 'E']:
        return text.strip()

    return None # 어떤 패턴도 찾지 못한 경우 None 반환


def compute_accuracy(output, tokenizer):
    print("--- compute_accuracy 함수 시작 ---")
    logits = output.predictions[0]
    predicted_token_ids = np.argmax(logits, axis=-1)
    pred_texts = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)

    label_ids_processed = np.where(output.label_ids == -100, tokenizer.pad_token_id, output.label_ids)
    label_texts = tokenizer.batch_decode(label_ids_processed, skip_special_tokens=True)
    
    # DEBUG
    print(f"\n디코딩된 예측 텍스트 샘플 (첫 3개):\n{pred_texts[:3]}")
    print(f"\n디코딩된 레이블 텍스트 샘플 (첫 3개):\n{label_texts[:3]}")

    assert len(pred_texts) == len(label_texts), "Prediction and label counts must match"

    # 3. 정확도 계산
    correct, total = 0, 0
    for p_text, a_text in zip(pred_texts, label_texts): # 이미 디코딩된 텍스트를 사용
        p_ans = extract_answer_letter(p_text)
        a_ans = extract_answer_letter(a_text)

        if not a_ans:
            print(f"경고: 유효하지 않은 정답 레이블이 디코딩되었습니다: '{a_text}'")
            continue
        
        total += 1
        if p_ans and p_ans == a_ans:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    print(f"\n정확도 계산 결과:")
    print(f"정확도: {accuracy:.4f}")
    print(f"전체 예측 수: {total}")
    print(f"정답 예측 수: {correct}")
    print("--- compute_accuracy 함수 종료 ---")

    return {
        "Test_Accuracy": round(accuracy, 4),
        "Total_preds": total,
        "Correct_preds": correct
    }


def main():
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
    
    ##--------------------------------------- Curriculum Learning --------------------------------
    # --------------------- DACON 데이터셋으로 TEST --------------------
    test_ds = load_dataset("json", data_files="eg/train_aug.jsonl", split='train')
    test_ds = DACONDataset(test_ds)
    
    # --------------------- Visual 7w -------------------------------
    # Visual 7w 데이터셋 로드
    ds = load_dataset("json", data_files="datasets/visual7w/visual7w_augmented_with_descriptions.jsonl", split='train')
    train_ds, val_ds = preproc_visual7w(ds)
    train_ds = VisualDataset(train_ds)
    val_ds = VisualDataset(val_ds)
    print(f"Total number of train samples: {len(train_ds)}")
    print(f"Total number of validation samples: {len(val_ds)}")
    
    # 메모리 정리
    del ds
    torch.cuda.empty_cache()
    gc.collect()
    
        # --------------------- Training Config -----------------------
    batch_size = 16
    training_args = SFTConfig(
        report_to='none',
        # run_name = 'run-0730-SOTA' + "-T5",
        output_dir="Model/flan-t5/sota",
        overwrite_output_dir=True,
        num_train_epochs=5,     # epochs
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=128 // batch_size,
        eval_accumulation_steps=4,
        warmup_ratio=0.05,
        learning_rate=2e-6,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy="best",
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
        eval_do_concat_batches=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_accuracy(eval_pred, tokenizer)
    )
    
    print("✅ Visual 7w Training start.")
    trainer.train()
    print("✅ Visual 7w Training done.")
    
    # 훈련 결과 저장
    final_save_directory = os.path.join(training_args.output_dir, "best")
    os.makedirs(final_save_directory, exist_ok=True)
    trainer.model.save_pretrained(final_save_directory)     # 모델 저장
    print(f"✅ Model saved to {final_save_directory}")
    tokenizer.save_pretrained(final_save_directory)     # tokenizer 저장
    print(f"✅ Tokenizer saved to {final_save_directory}")
    
    trainer.evaluate(test_ds, metric_key_prefix="test")   # SOTA는 60개 중 47개를 맞춘 스펙.
    

    # --------------------- Realworld QA -------------------------------
    # Realworld QA 데이터셋 로드
    ds = load_from_disk("datasets/Realworld/Realworld_aug")
    ds = ds.filter(lambda x: x['answer'] in ['A', 'B', 'C', 'D'])
    ds = ds.class_encode_column("answer").train_test_split(test_size=min(500, len(ds)//5), stratify_by_column='answer')
    train_ds = RealworldDataset(ds['train'])
    val_ds = RealworldDataset(ds['test'])
    
    # 메모리 정리
    del ds
    torch.cuda.empty_cache()
    gc.collect()
    
    training_args.num_train_epochs = 4
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_accuracy(eval_pred, tokenizer)
    )
    
    print("✅ Realworld QA Training start.")
    trainer.train()
    print("✅ Realworld QA Training done.")
    
    # 훈련 결과 저장
    final_save_directory = os.path.join(training_args.output_dir, "best")
    os.makedirs(final_save_directory, exist_ok=True)
    trainer.model.save_pretrained(final_save_directory)     # 모델 저장
    print(f"✅ Model saved to {final_save_directory}")
    tokenizer.save_pretrained(final_save_directory)     # tokenizer 저장
    print(f"✅ Tokenizer saved to {final_save_directory}")
    
    trainer.evaluate(test_ds, metric_key_prefix="test")   # SOTA는 60개 중 47개를 맞춘 스펙.
    
    
    # --------------------- A-OKVQA -------------------------------
    train_ds = load_dataset("HuggingFaceM4/A-OKVQA", split="train")
    val_ds = load_dataset("HuggingFaceM4/A-OKVQA", split="validation")
    val_ds = val_ds.shuffle().select(range(200))
    train_ds = AokvqaDataset(train_ds)
    val_ds = AokvqaDataset(val_ds)
    
    # 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()
    
    training_args.num_train_epochs = 3
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_accuracy(eval_pred, tokenizer)
    )
    
    print("✅ A-OKVQA Training start.")
    trainer.train()
    print("✅ A-OKVQA Training done.")
    
    # 훈련 결과 저장
    final_save_directory = os.path.join(training_args.output_dir, "best")
    os.makedirs(final_save_directory, exist_ok=True)
    trainer.model.save_pretrained(final_save_directory)     # 모델 저장
    print(f"✅ Model saved to {final_save_directory}")
    tokenizer.save_pretrained(final_save_directory)     # tokenizer 저장
    print(f"✅ Tokenizer saved to {final_save_directory}")
    
    trainer.evaluate(test_ds, metric_key_prefix="test")   # SOTA는 60개 중 47개를 맞춘 스펙.

 # --------------------- VMC Dataset -------------------------------
    ds = load_from_disk("datasets/VMC/VMC_aug")
    class_labels = ClassLabel(names=["A", "B", "C", "D"])
    ds = ds.cast_column("answer", class_labels)
    ds = ds.train_test_split(test_size=min(200, len(ds)//5), stratify_by_column='answer')
    train_ds, val_ds = ds['train'], ds['test']
    train_ds = VMCDataset(train_ds)
    val_ds = VMCDataset(val_ds)
    
    # 메모리 정리
    del ds
    torch.cuda.empty_cache()
    gc.collect()
    
    training_args.num_train_epochs = 3
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_accuracy(eval_pred, tokenizer)
    )
    
    print("✅ VMC Training start.")
    trainer.train()
    print("✅ VMC Training done.")
    
    # 훈련 결과 저장
    final_save_directory = os.path.join(training_args.output_dir, "best")
    os.makedirs(final_save_directory, exist_ok=True)
    trainer.model.save_pretrained(final_save_directory)     # 모델 저장
    print(f"✅ Model saved to {final_save_directory}")
    tokenizer.save_pretrained(final_save_directory)     # tokenizer 저장
    print(f"✅ Tokenizer saved to {final_save_directory}")
    
    trainer.evaluate(test_ds, metric_key_prefix="test")   # SOTA는 60개 중 47개를 맞춘 스펙.
    
    print("✅ All Training Done.")
    
    # 모델 정보
    total_params = sum(p.numel() for p in model.parameters())
    print(f"모델 정보:")
    print(f"총 파라미터 수: {total_params:,}")


if __name__ == "__main__":
    main()