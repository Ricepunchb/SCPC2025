import os
import re
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (InstructBlipForConditionalGeneration,
                          T5ForConditionalGeneration,
                          InstructBlipConfig,
                          InstructBlipProcessor, 
                          InstructBlipVisionConfig, 
                          InstructBlipQFormerConfig, 
                          T5Config,
                          T5Tokenizer)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

from customdatasets import AokvqaDataset, DaconDataset, collate_fn



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


# 1. 기존 모델 config 로드
vision_config = InstructBlipVisionConfig.from_pretrained("Salesforce/instructblip-flan-t5-xl")
qformer_config = InstructBlipQFormerConfig.from_pretrained("Salesforce/instructblip-flan-t5-xl")
text_config = T5Config.from_pretrained("google/flan-t5-large")

# T5 config 조정 (InstructBLIP에 맞게)
text_config.is_encoder_decoder = True
text_config.use_cache = True
if not hasattr(text_config, 'bos_token_id') or text_config.bos_token_id is None:
    text_config.bos_token_id = 1  # T5의 경우 보통 0

# 2. 새로운 config로 InstructBLIP 모델 생성
config = InstructBlipConfig.from_vision_qformer_text_configs(
    vision_config, qformer_config, text_config
)

# 3. 빈 모델 초기화
model = InstructBlipForConditionalGeneration(config)

print("모델 구조 초기화 완료")
print(f"Vision model hidden size: {config.vision_config.hidden_size}")
print(f"Q-Former hidden size: {config.qformer_config.hidden_size}")
print(f"Text model hidden size: {config.text_config.hidden_size}")

# 4. 사전 훈련된 컴포넌트들 로드
print("사전 훈련된 가중치 로드 중...")

# 전체 원본 InstructBLIP 모델 로드 (vision과 qformer 가중치 추출용)
original_model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")

# Vision model 가중치 복사
model.vision_model.load_state_dict(original_model.vision_model.state_dict())
print("✓ Vision model 가중치 로드 완료")

# Q-Former 가중치 복사
model.qformer.load_state_dict(original_model.qformer.state_dict())
print("✓ Q-Former 가중치 로드 완료")

# Language model 로드 (flan-t5-large)
new_lm = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
model.language_model.load_state_dict(new_lm.state_dict())
print("✓ Language model (flan-t5-large) 가중치 로드 완료")

# 메모리 정리
del original_model, new_lm
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# 5. Language projection layer 재초기화
# Q-Former의 출력을 새로운 language model의 입력 차원에 맞게 조정
qformer_hidden_size = config.qformer_config.hidden_size  # 768
text_hidden_size = config.text_config.hidden_size        # 1024 (flan-t5-large)

# 기존 projection layer를 새로운 차원에 맞게 교체
model.language_projection = nn.Linear(
    qformer_hidden_size, 
    text_hidden_size, 
    bias=True
)

# Xavier uniform 초기화로 안정적인 학습 시작
nn.init.xavier_uniform_(model.language_projection.weight)
print(f"✓ Language projection layer 재초기화 완료: {qformer_hidden_size} -> {text_hidden_size}")

# 6. 모델 검증
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n모델 정보:")
print(f"총 파라미터 수: {total_params:,}")




# 기본적으로 기존 InstructBLIP processor 사용
# tokenizer는 자동으로 flan-t5-large에 맞게 조정됨
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
xl_tokens = set(processor.tokenizer.get_vocab().keys())     # 원래 토크나이저에서 vocab 추출

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
large_tokens = set(tokenizer.get_vocab().keys())     # 바꿀 토크나이저가 원래 갖고있던 vocab 추출
tokens_to_add = list(xl_tokens - large_tokens)      # 추가로 계승할 토큰
num_added = tokenizer.add_tokens(tokens_to_add)   # 3) flan‑t5‑large 토크나이저에 추가
print(f"Added {num_added} tokens from flan-t5-xl into flan-t5-large tokenizer")

processor.tokenizer = tokenizer
model.resize_token_embeddings(len(tokenizer))  # LM head(임베딩) 크기 조정
model.language_model.resize_token_embeddings(len(tokenizer))
model.config.text_config.vocab_size = len(tokenizer)
model.config.vocab_size = len(tokenizer)
del tokenizer   # 메모리 정리


# PEFT-LoRA 적용
target_module_names = []
for name, module in model.named_modules():
    if ("qformer" in name or "language_model" in name) and isinstance(module, (nn.Linear, nn.Embedding)):   # Qformer와 langauge모델만 훈련
        target_module_names.append(name.split('.')[-1]) 

# 중복 제거
target_module_names = list(set(target_module_names))

lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    lora_dropout=0.05,
    bias='none',
    target_modules=target_module_names        
)
model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())



# A-OKVQA로 파인튜닝
batch_size = 2

train_ds = load_dataset("HuggingFaceM4/A-OKVQA", split="train")
val_ds = load_dataset("HuggingFaceM4/A-OKVQA", split="validation")
train_ds = AokvqaDataset(dataset=train_ds, processor=processor)
val_ds = AokvqaDataset(dataset=val_ds, processor=processor)
train_dataloader = DataLoader(train_ds, batch_size=batch_size, collate_fn = collate_fn, shuffle=True, pin_memory=True, num_workers=4)
valid_dataloader = DataLoader(val_ds, batch_size=batch_size, collate_fn = collate_fn, pin_memory=True, num_workers=4)

# --- 1번째 훈련 루프 ---
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
# num_epochs = 10
num_epochs = 1
patience = 3
min_eval_loss = float("inf")
early_stopping_hook = 0
tracking_information = []
gradient_accumulation_steps = 128 // batch_size
model.to(device)
for epoch in range(num_epochs):
    # ========================= Training =========================
    model.train()
    
    step = 1
    epoch_loss = 0
    
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
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
        
        if step%gradient_accumulation_steps == 0 or step == len(train_dataloader):
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * gradient_accumulation_steps
        step += 1

    # ========================= Validation =========================
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):

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
    
    # 스케줄러 업데이트
    scheduler.step()

    # 조기 종료 및 모델 저장
    # Early stopping은 전체 loss 합이 아닌 평균 loss로 비교하는 것이 더 직관적입니다.
    if avg_eval_loss < min_eval_loss:
        min_eval_loss = avg_eval_loss
        early_stopping_hook = 0
        model.save_pretrained("Model/instructblip-lora")
        model.base_model.save_pretrained('Model/instructblip-base')
        processor.save_pretrained("Model/instructblip-processor")
        print(f"Validation loss decreased ({min_eval_loss:.4f}). Saving model...")
    else:
        early_stopping_hook += 1
        if early_stopping_hook >= patience:
            print(f"Early stopping at epoch {epoch+1} as validation loss did not improve for {patience} epochs.")
            break


# 중간 메모리 정리
del train_ds, train_dataloader
torch.cuda.empty_cache() if torch.cuda.is_available() else None


# DACON 에서 준 중요한 데이터셋
dacon_ds = load_dataset("csv",  data_files="./train.csv", split='train')
dacon_ds = DaconDataset(dataset=dacon_ds, processor=processor)
dacon_dataloader = DataLoader(dacon_ds, batch_size=batch_size, collate_fn = collate_fn, shuffle=True, pin_memory=True, num_workers=4)

# --- 2번째 훈련 루프 ---
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
# num_epochs = 3
num_epochs = 1
min_eval_loss = float("inf")
early_stopping_hook = 0
tracking_information = []
gradient_accumulation_steps = 128 // batch_size

for epoch in range(num_epochs):
    # ========================= Training =========================
    model.train()
    
    step = 1
    epoch_loss = 0
    
    for batch in tqdm(dacon_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
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
        
        if step % gradient_accumulation_steps == 0 or step == len(dacon_dataloader):
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * gradient_accumulation_steps
        step += 1

    # ========================= Validation =========================
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):

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
    avg_train_loss = epoch_loss / len(dacon_dataloader)
    avg_eval_loss = eval_loss / len(valid_dataloader)
    current_lr = optimizer.param_groups[0]["lr"]

    tracking_information.append((avg_train_loss, avg_eval_loss, current_lr))
    print(f"Epoch: {epoch+1} | Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_eval_loss:.4f} | LR: {current_lr}")
    
    # 스케줄러 업데이트
    scheduler.step()

    # 조기 종료 및 모델 저장
    # Early stopping은 전체 loss 합이 아닌 평균 loss로 비교하는 것이 더 직관적입니다.
    if avg_eval_loss < min_eval_loss:
        min_eval_loss = avg_eval_loss
        early_stopping_hook = 0
        model.save_pretrained("Model/instructblip-lora")
        model.base_model.save_pretrained('Model/instructblip-base')
        processor.save_pretrained("Model/instructblip-processor")
        print(f"Validation loss decreased ({min_eval_loss:.4f}). Saving model...")
        
        


# 추론
test = pd.read_csv('./test.csv')
results = []

# 정답 알파벳 추출 함수
def extract_answer_letter(text):
    match = re.search(r"\s*([A-Da-d])\b", text)
    return match.group(1).upper() if match else "?"


for _, row in tqdm(test.iterrows(), total=len(test)):
    image = Image.open(row['img_path']).convert("RGB")
    choices = [row[c] for c in ['A', 'B', 'C', 'D']]

    prompt = (
        "<image>\n"
        "Based on the image, choose the correct option to the following question.\n"
        f"Question: {row['Question']}\n"
        "Options:"
        + "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]) +
        "\nAnswer:"
    )

    inputs = processor(images=image, text=prompt, padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(device)

    output = model.generate(**inputs, 
                            num_beams=5, 
                            top_p=0.9, 
                            repetition_penalty=1.5, 
                            length_penalty=1.0, 
                            temperature=0.7, 
                            max_new_tokens=3, 
                            do_sample=False)
    decoded = processor.tokenizer.decode(output[0], skip_special_tokens=True).strip()
    print(decoded)
    results.append(extract_answer_letter(decoded))

print('✅ Inference Done.')

submission = pd.read_csv('./sample_submission.csv')
submission['answer'] = results
submission.to_csv('./baseline_submit.csv', index=False)
print("✅ CSV for submission Done.")