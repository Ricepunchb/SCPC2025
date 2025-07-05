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
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from transformers import (InstructBlipForConditionalGeneration,
                          T5ForConditionalGeneration,
                          InstructBlipConfig,
                          InstructBlipProcessor, 
                          InstructBlipVisionConfig, 
                          InstructBlipQFormerConfig, 
                          T5Config,
                          T5Tokenizer,
                          AddedToken,
                          get_cosine_schedule_with_warmup)
from datasets import load_dataset, Dataset
from peft import LoraConfig, PeftModel, get_peft_model

from customdatasets import AokvqaDataset, DaconDataset, VisualDataset, collate_fn


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

# --------------------- 모델 선언부 -----------------------
# 기존 모델 config 로드
vision_config = InstructBlipVisionConfig.from_pretrained("Salesforce/instructblip-flan-t5-xl")
qformer_config = InstructBlipQFormerConfig.from_pretrained("Salesforce/instructblip-flan-t5-xl")
text_config = T5Config.from_pretrained("google/flan-t5-large")

# T5 config 조정 (InstructBLIP에 맞게)
text_config.is_encoder_decoder = True
text_config.use_cache = True
if not hasattr(text_config, 'bos_token_id') or text_config.bos_token_id is None:
    text_config.bos_token_id = 1

# config로 InstructBLIP 모델 생성
config = InstructBlipConfig.from_vision_qformer_text_configs(
    vision_config, qformer_config, text_config
)

# 빈 모델 초기화
model = InstructBlipForConditionalGeneration(config)

print("Model architecture initialized.")

# 사전 훈련된 컴포넌트들 로드
print("Loading pretrained checkpoints...")
# 원본 InstructBLIP 모델 로드 (vision과 qformer 가중치 추출용)
original_model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")

# Vision model 가중치 계승
model.vision_model.load_state_dict(original_model.vision_model.state_dict())
print("✓ Vision model checkpoint loaded")

# Q-Former 가중치 계승
model.qformer.load_state_dict(original_model.qformer.state_dict())
print("✓ Q-Former checkpoint loaded")

# LM (flan-t5-large) 가중치 계승 
new_lm = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
model.language_model.load_state_dict(new_lm.state_dict())
print("✓ Language model (flan-t5-large) checkpoint loaded")

# 메모리 정리
del original_model, new_lm
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# projection layer를 Xavier uniform 초기화
nn.init.xavier_uniform_(model.language_projection.weight)
print(f"✓ Language projection layer 재초기화 완료")

# 모델 정보
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n모델 정보:")
print(f"총 파라미터 수: {total_params:,}")


# --------------------- processor, tokenizer 편집부 -----------------------
# # 원래 토크나이저에서 vocab 추출
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")

# <image> 토큰 확인
def check(tokenizer):
    image_token_id = None
    # 토크나이저에 <image> 토큰이 이미 있는지 확인
    if "<image>" in tokenizer.get_vocab():
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        print(f"'<image>' 토큰이 토크나이저 어휘에 이미 존재합니다. 인덱스: {image_token_id}")
        # 해당 토큰이 special token인지 확인
        if image_token_id in tokenizer.all_special_tokens:
            print(f"'{image_token_id}' 토큰은 스페셜 토큰입니다.")
        else:
            print(f"'{image_token_id}' 토큰은 일반 토큰입니다 (하지만 모델 내부적으로는 특별하게 처리됨).")
    else: print("!! <image> 토큰이 없음!!")
    return image_token_id

print('large의 경우: \n',check(tokenizer))
print('기존 xl의 경우: \n',check(processor.tokenizer))

xl_tokens = set(processor.tokenizer.get_vocab().keys())     # 기존. xl 모델의 토큰들
large_tokens = set(tokenizer.get_vocab().keys())     # large 모델의 토큰들

image_token = AddedToken("<image>", normalized=False, special=True)
tokenizer.add_tokens([image_token], special_tokens=True)
print(f"Added {image_token} to T5-Large tokenizer.")

tokens_to_add = list(xl_tokens - large_tokens)      # 추가 계승할 토큰
print("More tokens to success: ",tokens_to_add)

num_added = tokenizer.add_tokens(tokens_to_add)   # flan‑t5‑large 토크나이저에 추가
print(f"Added {num_added} tokens from flan-t5-xl into flan-t5-large tokenizer")

processor.tokenizer = tokenizer     # 토크나이저 계승

model.language_model.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=64)    # LM head(임베딩) 크기 리사이즈
print(f"Language model embedding size resized to {len(processor.tokenizer)}.")

image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")   # 토크나이저의 <image> 토큰 인덱스
model.config.text_config.image_token_index = image_token_id
model.config.image_token_index = image_token_id
print(model.config.image_token_index, model.config.text_config.image_token_index)       # <image> 토큰 인덱스 위랑 확인, 이 두 숫자 같은지 확인

model.config.text_config.vocab_size = len(processor.tokenizer)      # vocab size 업데이트
model.config.vocab_size = len(processor.tokenizer)

del tokenizer, xl_tokens, large_tokens   # 메모리 정리


# 베이스 모델 저장. !!반드시 PEFT 적용전에 해야함!!
model.base_model.save_pretrained('./Model/instructblip-base')


# -------------------------PEFT LoRA 설정부----------------------------
# PEFT-LoRA 적용
target_module_names = [
    # Q-Former 내의 선형 레이어 (BERT-like 구조)
    "query",            # Q-Former 어텐션의 쿼리 프로젝션
    "key",              # Q-Former 어텐션의 키 프로젝션
    "value",            # Q-Former 어텐션의 값 프로젝션
    "attention.output.dense", # Q-Former 어텐션의 출력 밀집 레이어
    "intermediate.dense", # Q-Former FFN의 중간 밀집 레이어
    "output.dense",       # Q-Former FFN의 출력 밀집 레이어

    # Language Model (Flan-T5-large) 내의 선형 레이어
    "q",                # T5 어텐션의 쿼리 프로젝션
    "k",                # T5 어텐션의 키 프로젝션
    "v",                # T5 어텐션의 값 프로젝션
    "o",                # T5 어텐션의 출력 프로젝션
    "wi_0",             # T5 FFN의 중간 레이어 1
    "wi_1",             # T5 FFN의 중간 레이어 2 (Gated FFN의 경우)
    "wo",               # T5 FFN의 출력 레이어

    # InstructBLIP 특정 레이어
    "language_projection", # Q-Former -> LLM 연결 projection
    "lm_head"             # 언어 모델의 최종 출력 헤드
]

lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    lora_dropout=0.05,
    bias='none',
    target_modules=target_module_names        
)
model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())

del target_module_names     # 중간 메모리 정리


# # ----------------------------------- [Optional] 모델 로드 ------------------------------------
# processor = InstructBlipProcessor.from_pretrained("./Model/instructblip-processor")
# model = InstructBlipForConditionalGeneration.from_pretrained('./Model/instructblip-base')

# peft_model_path = "./Model/instructblip-lora"
# model = PeftModel.from_pretrained(model, peft_model_path)
# model.to('cuda')


# ---------------------------------------- 훈련부 ---------------------------------------- 
# WandB 시동
wandb.init(project="SCPC 2025")
wandb.watch(model, log="all", log_freq=50)
wandb.config.update({"nparams": sum([p.numel() for p in model.parameters() if p.requires_grad])})

batch_size = 2

# -------------------------------------- A-OKVQA 훈련 --------------------------------------------
train_ds = load_dataset("HuggingFaceM4/A-OKVQA", split="train")
val_ds = load_dataset("HuggingFaceM4/A-OKVQA", split="validation")
train_ds = AokvqaDataset(dataset=train_ds, processor=processor)
val_ds = AokvqaDataset(dataset=val_ds, processor=processor)
train_dataloader = DataLoader(train_ds, batch_size=batch_size, collate_fn = collate_fn, shuffle=True, pin_memory=True, num_workers=4)
valid_dataloader = DataLoader(val_ds, batch_size=batch_size, collate_fn = collate_fn, pin_memory=True, num_workers=4)

# --- 1번째 훈련 루프 ---
num_epochs = 5 
patience = 2
min_eval_loss = float("inf")
early_stopping_hook = 0
tracking_information = []
gradient_accumulation_steps = 128 // batch_size

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
num_training_steps = int( len(train_dataloader) * num_epochs )                      
scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=int(num_training_steps * 0.01),
    num_training_steps=num_training_steps
    )

model.to(device)

step = 1
for epoch in range(num_epochs):
    # ========================= Training =========================
    model.train()
    
    epoch_loss = 0
    
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", dynamic_ncols=True):
        pixel_values = batch.pop("pixel_values").to(device)
        input_ids = batch.pop("input_ids").to(device)
        qformer_input_ids = batch.pop('qformer_input_ids').to(device)
        qformer_attention_mask = batch.pop('qformer_attention_mask').to(device)
        labels = batch.pop('labels').to(device)

        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            qformer_input_ids=qformer_input_ids,
            qformer_attention_mask=qformer_attention_mask,
            labels=labels
            )
        loss = outputs.loss
        loss /= gradient_accumulation_steps

        # 역전파
        loss.backward()
        
        if step % gradient_accumulation_steps == 0 or step == len(train_dataloader):
            wandb.log({'A-OKVQA/current_avg_loss': loss * gradient_accumulation_steps}, step=step)
            optimizer.step()
            scheduler.step()
            
            optimizer.zero_grad()            
        
        epoch_loss += loss.item() * gradient_accumulation_steps
        
        step += 1

    # ========================= Validation =========================
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", dynamic_ncols=True):

            pixel_values = batch.pop("pixel_values").to(device)
            input_ids = batch.pop("input_ids").to(device)
            qformer_input_ids = batch.pop('qformer_input_ids').to(device)
            qformer_attention_mask = batch.pop('qformer_attention_mask').to(device)
            labels = batch.pop('labels').to(device)
    
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                qformer_input_ids=qformer_input_ids,
                qformer_attention_mask=qformer_attention_mask,
                labels=labels
                )
            loss = outputs.loss
            
            eval_loss += loss.item()


    # --- 에포크 마무리 및 로깅 ---
    avg_train_loss = epoch_loss / len(train_dataloader)
    avg_eval_loss = eval_loss / len(valid_dataloader)
    current_lr = optimizer.param_groups[0]["lr"]

    tracking_information.append((avg_train_loss, avg_eval_loss, current_lr))
    print(f"Epoch: {epoch+1} | Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_eval_loss:.4f} | LR: {current_lr}")
    
    # WandB 로깅
    wandb.log({"A-OKVQA/epoch/train_loss": avg_train_loss,
               "A-OKVQA/epoch/val_loss": avg_eval_loss,
               "A-OKVQA/LR": current_lr,
               "A-OKVQA/epoch": epoch+1},
              step=step)
    

    # 조기 종료 및 모델 저장
    # Early stopping은 평균 loss로 비교하는 것이 더 직관적
    if avg_eval_loss < min_eval_loss:
        min_eval_loss = avg_eval_loss
        early_stopping_hook = 0
        model.save_pretrained("Model/instructblip-lora")
        processor.save_pretrained("Model/instructblip-processor")
        print(f"Validation loss decreased ({min_eval_loss:.4f}). Saving model...")
    else:
        early_stopping_hook += 1
        if early_stopping_hook >= patience:
            print(f"Early stopping at epoch {epoch+1} as validation loss did not improve for {patience} epochs.")
            break



# 중간 메모리 정리
del train_ds, train_dataloader, val_ds, valid_dataloader
torch.cuda.empty_cache() if torch.cuda.is_available() else None


# -------------------------------------- Visual 7w 훈련 -----------------------------------------
# Visual 7w 데이터셋 로드
visual_ds = load_dataset("json", data_files="/mnt/workspace/datasets/visual7w/dataset_v7w_telling.json", split='train')
# 데이터셋 전처리
processed = []
for sample in visual_ds:
    sample = sample['images']
    image_name = sample['filename']
    
    for i in range(len(sample['qa_pairs'])):
        answer_idx = np.random.randint(0,4)
        choices = sample['qa_pairs'][i]['multiple_choices']
        choices.insert(answer_idx, sample['qa_pairs'][i]['answer'])
        processed.append({'question': sample['qa_pairs'][i]['question'],
                          'answer': sample['qa_pairs'][i]['answer'],
                          'answer_idx': answer_idx,
                          'choices': choices,
                          'img_path': f'datasets/visual7w/images/{image_name}'   })

del visual_ds   # 중간 메모리 정리

processed = Dataset.from_list(processed)
processed = processed.class_encode_column('answer_idx').train_test_split(test_size=0.2, stratify_by_column='answer_idx')
train_ds, val_ds = processed['train'], processed['test']
train_ds = VisualDataset(dataset=train_ds, processor=processor)
val_ds = VisualDataset(dataset=val_ds, processor=processor)
train_dataloader = DataLoader(train_ds, batch_size=batch_size, collate_fn = collate_fn, shuffle=True, pin_memory=True, num_workers=4)
valid_dataloader = DataLoader(val_ds, batch_size=batch_size, collate_fn = collate_fn, shuffle=False, pin_memory=True, num_workers=4)

del processed   # 중간 메모리 정리

# --- 2번째 훈련 루프 ---
num_epochs = 5
patience = 2
min_eval_loss = float("inf")
early_stopping_hook = 0
tracking_information = []
gradient_accumulation_steps = 128 // batch_size

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)
num_training_steps = int(len(train_dataloader * num_epochs))                      
scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=int(num_training_steps * 0.01),
    num_training_steps=num_training_steps
    )

model.to(device)

step = 1
for epoch in range(num_epochs):
    # ========================= Training =========================
    model.train()
    
    epoch_loss = 0
    
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", dynamic_ncols=True):
        pixel_values = batch.pop("pixel_values").to(device)
        input_ids = batch.pop("input_ids").to(device)
        qformer_input_ids = batch.pop('qformer_input_ids').to(device)
        qformer_attention_mask = batch.pop('qformer_attention_mask').to(device)
        labels = batch.pop('labels').to(device)

        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            qformer_input_ids=qformer_input_ids,
            qformer_attention_mask=qformer_attention_mask,
            labels=labels
            )
        loss = outputs.loss
        loss /= gradient_accumulation_steps

        loss.backward()
        
        if step % gradient_accumulation_steps == 0 or step == len(train_dataloader):
            wandb.log({'Visual7w/current_avg_loss': loss * gradient_accumulation_steps}, step=step)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * gradient_accumulation_steps
        
        step += 1

    # ========================= Validation =========================
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", dynamic_ncols=True):

            pixel_values = batch.pop("pixel_values").to(device)
            input_ids = batch.pop("input_ids").to(device)
            qformer_input_ids = batch.pop('qformer_input_ids').to(device)
            qformer_attention_mask = batch.pop('qformer_attention_mask').to(device)
            labels = batch.pop('labels').to(device)
    
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                qformer_input_ids=qformer_input_ids,
                qformer_attention_mask=qformer_attention_mask,
                labels=labels
                )
            loss = outputs.loss
            
            eval_loss += loss.item()


    # --- 에포크 마무리 및 로깅 ---
    avg_train_loss = epoch_loss / len(train_dataloader)
    avg_eval_loss = eval_loss / len(valid_dataloader)
    current_lr = optimizer.param_groups[0]["lr"]

    tracking_information.append((avg_train_loss, avg_eval_loss, current_lr))
    print(f"Epoch: {epoch+1} | Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_eval_loss:.4f} | LR: {current_lr}")
    
    # WandB 로깅
    wandb.log({"Visual7w/epoch/train_loss": avg_train_loss,
               "Visual7w/epoch/val_loss": avg_eval_loss,
               "Visual7w/LR": current_lr,
               "Visual7w/epoch": epoch+1 },
              step=step)
    

    # 조기 종료 및 모델 저장
    if avg_eval_loss < min_eval_loss:
        min_eval_loss = avg_eval_loss
        early_stopping_hook = 0
        model.save_pretrained("Model/instructblip-lora")
        processor.save_pretrained("Model/instructblip-processor")
        print(f"Validation loss decreased ({min_eval_loss:.4f}). Saving model...")
    else:
        early_stopping_hook += 1
        if early_stopping_hook >= patience:
            print(f"Early stopping at epoch {epoch+1} as validation loss did not improve for {patience} epochs.")
            break


# 중간 메모리 정리
del train_ds, val_ds, train_dataloader, valid_dataloader
torch.cuda.empty_cache() if torch.cuda.is_available() else None


# DACON 에서 준 중요한 데이터셋
dacon_ds = load_dataset("csv",  data_files="eg/train.csv", split='train')
indices = list(range(len(dacon_ds)))        # 전체 데이터 인덱스 및 라벨 추출
labels = [dacon_ds[i]['answer'] for i in indices]  # A/B/C/D 라벨


# -------------------------------------- DACON 공식 데이터셋 훈련 -----------------------------------------
# Stratified K-Fold 설정
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

# 학습 설정
num_epochs = 3
patience = 1

model.to(device)

fold_results = []

step = 1
for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels), start=1):
    print(f"\n===== Fold {fold}/{n_splits} =====")
    # Subset 및 DataLoader
    train_subset = Subset(dacon_ds, train_idx.tolist())
    val_subset = Subset(dacon_ds, val_idx.tolist())
    train_subset = DaconDataset(dataset=train_subset, processor=processor)
    val_subset = DaconDataset(dataset=val_subset, processor=processor)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=4)

    # Optimizer 및 Scheduler 초기화
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        # ----- Training -----
        model.train()
        optimizer.zero_grad()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch} [Train]", dynamic_ncols=True):
            # Move to device
            for k, v in batch.items():
                batch[k] = v.to(device)

            outputs = model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                qformer_input_ids=batch['qformer_input_ids'],
                qformer_attention_mask=batch['qformer_attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss 
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            step += 1

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ----- Validation -----
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Fold {fold} Epoch {epoch} [Val]", dynamic_ncols=True):
                for k, v in batch.items():
                    batch[k] = v.to(device)
                outputs = model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    qformer_input_ids=batch['qformer_input_ids'],
                    qformer_attention_mask=batch['qformer_attention_mask'],
                    labels=batch['labels']
                )
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Fold {fold} Epoch {epoch} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {current_lr:.1e}")
        
        # WandB 로깅
        wandb.log({"Official/epoch/train_loss": avg_train_loss,
                   "Official/epoch/val_loss": avg_val_loss,
                   "Official/LR": current_lr,
                   "Official/epoch": epoch  },
                  step=step)

        
        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 체크포인트 저장
            model.save_pretrained('./Model/instructblip-lora')
            processor.save_pretrained('./Model/instructblip-processor')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} for fold {fold}")
                break

    fold_results.append(best_val_loss)

# K-Fold 결과 요약
print("\n===== K-Fold Summary =====")
for i, loss in enumerate(fold_results, start=1):
    print(f"Fold {i} Best Val Loss: {loss:.4f}")
print(f"Mean Val Loss: {sum(fold_results)/len(fold_results):.4f}")
        
        

# 추론
test = pd.read_csv('eg/test.csv')
results = []

# 정답 알파벳 추출 함수
def extract_answer_letter(text):
    match = re.search(r"\s*([A-Da-d])\.\s*", text)
    return match.group(1).upper() if match else "?"


for _, row in tqdm(test.iterrows(), total=len(test)):
    image = Image.open('eg/'+row['img_path']).convert("RGB")
    choices = [row[c] for c in ['A', 'B', 'C', 'D']]

    prompt = (
        "Based on the image, choose the correct option to the following question.\n"
        f"Question: {row['Question']}\n"
        "Options:"
        + "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]) +
        "\nAnswer:"
    )

    inputs = processor(images=image, text=prompt, padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        do_sample=False,       # 샘플링 비활성화 (결정론적 빔 서치)
        num_beams=8,           # 8개의 빔을 사용하여 탐색
        early_stopping=True,   # 빔이 EOS에 도달하면 일찍 중지
        max_new_tokens=50,     # 생성할 최대 새 토큰 수
        min_length=1,
        repetition_penalty=1.2,
        length_penalty=0.8,    # 적절한 길이의 답변 유도
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