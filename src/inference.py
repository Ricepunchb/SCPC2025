import os
import re
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from collections import Counter
from datasets import load_dataset
import torch

from transformers import AutoProcessor, Kosmos2ForConditionalGeneration, T5ForConditionalGeneration, T5Tokenizer, BitsAndBytesConfig
from peft import PeftModel

from torch.utils.data import ConcatDataset, DataLoader

from customdatasets import *


# 정답 알파벳 추출 함수
def extract_answer_letter(text):
    """
    주어진 텍스트에서 최종 답변 알파벳 (예: '(A)')을 추출합니다.
    모델의 출력 패턴에 따라 정규 표현식 등을 사용하여 더 견고하게 만들 수 있습니다.
    """
    # 괄호와 점이 포함된 패턴: "The answer is: (A)." 또는 "Answer: (A)."
    match = re.search(r'(?:The answer is:|Answer:)\s*\((\s*[A-D]\s*)\)\.?', text)
    if match:
        return match.group(1).strip()

    # 괄호만 포함된 패턴: "The answer is: (A)" 또는 "Answer: (A)"
    match_paren_only = re.search(r'(?:The answer is:|Answer:)\s*\((\s*[A-D]\s*)\)', text)
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

    # 모델이 단순히 'A' 또는 'B'만 생성했을 때를 대비.
    # 예: text = "A"
    if text.strip() in ['A', 'B', 'C', 'D']:
        return text.strip()

    return "X" # 어떤 패턴도 찾지 못한 경우 'X' 반환


def extract_text_after_prefix(full_text, prefix="Describe this image in detail."):
    """
    주어진 텍스트에서 특정 접두사 뒤에 오는 모든 텍스트를 추출
    """
    pattern = re.compile(re.escape(prefix) + r'(.*)', re.DOTALL)
    
    match = pattern.search(full_text)
    if match:
        return match.group(1).strip()
    return None


def extract_description_content(full_text):
    """
    'Describe this image in detail.' 뒤의 텍스트를 파싱한 후,
    그 결과 텍스트 내에서 'Description: ' 뒤의 내용만을 다시 파싱합니다.
    """
    first_parsed_text = extract_text_after_prefix(full_text)
    
    if first_parsed_text is None:
        return "No Description." # 파싱된게 없으면..

    description_pattern = re.compile(r'Description:\s*(.*)', re.DOTALL)
    
    match = description_pattern.search(first_parsed_text)
    if match:
        return match.group(1).strip() # 추출된 텍스트의 앞뒤 공백 제거
    
    return first_parsed_text


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
    print("✅ Seed fixed")
    
    # ------------------------------ Qunatization Config -----------------------------------
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,                   # 모델을 4비트로 로드
        bnb_4bit_quant_type="nf4",             # QLoRA의 NF4
        bnb_4bit_compute_dtype=torch.bfloat16, # 연산 시 bfloat16 사용
        )
    
    # flan-t5-large 로드
    t5_directory = os.path.join("Model/flan-t5/sota/best")
    flan_t5 = T5ForConditionalGeneration.from_pretrained(t5_directory, device_map="auto", attn_implementation='eager')   # 모델 로드
    tokenizer = T5Tokenizer.from_pretrained(t5_directory)
    print("✅ Flan-T5-Large loaded")
    
    # KOSMOS2 로드
    kosmos_directory = os.path.join("Model/kosmos2/sota/phase_2/best")
    kosmos = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224", 
                                                             quantization_config=quantization_config, 
                                                             torch_dtype=torch.bfloat16, 
                                                             device_map="auto", 
                                                             attn_implementation="sdpa")   # 베이스모델 로드
    kosmos = PeftModel.from_pretrained(kosmos, kosmos_directory, torch_dtype=torch.bfloat16, device_map="auto")        # LoRA weight 적용된 모델 로드
    processor = AutoProcessor.from_pretrained(kosmos_directory, use_fast=True)
    print("✅ KOSMOS-2 loaded")
    
    if hasattr(torch, 'compile'):       # 추론 속도 최적화를 위한 torch compile
        print("Compiling models with torch.compile...")
        kosmos = torch.compile(kosmos)
        flan_t5 = torch.compile(flan_t5)
        print("Models compiled.")
    else:
        print("torch.compile is not available. Consider upgrading to PyTorch 2.0+ for potential speedups.")
    
    
    # ----------------------------------------------- 제출용 추론 부분 --------------------------------------------------
    test = pd.read_csv('eg/test.csv')
    
    results = []
    
    kosmos.eval()
    flan_t5.eval()
    
    print("✅ Inference Start.")
    
    for _, row in tqdm(test.iterrows(), total=len(test)):
        image = Image.open('eg/'+row['img_path']).convert("RGB")
        choices = '\n'.join([f"{c}. {row[c]}" for c in ['A', 'B', 'C', 'D']])

        prompt = f"Describe this image in detail."
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = kosmos.generate(                                                       # description 생성
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                image_embeds=None,
                use_cache=True,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.5,
                min_new_tokens=25,      
                max_new_tokens=1024,
                # num_beams
                )
        
        generated_text = processor.batch_decode(output, skip_special_tokens=True)[0]
        description, entities = processor.post_process_generation(generated_text)
        description = extract_description_content(description)
        print(f"Generated:\n {description}\n {row['Question']}\n {choices}")    # Image description부터 생성 및 확인

        # flan-t5-large로 최종 답안 도출
        prompt = f'Read the context and answer the question by choosing the right option among A, B, C and D by reasoning.\n Question: {row["Question"]}\n Choices: {choices}\n Description: {description}'
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
        output = flan_t5.generate(
                    **inputs,
                    use_cache=True,
                    do_sample=False,
                    # top_k=50,
                    # top_p=0.95,
                    # temperature=0.8,
                    num_beams=3,            # 3빔 서치
                    # num_return_sequences=1,
                    # max_new_tokens = 2048,
                    # no_repeat_ngram_size = 4
                    )
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        final_answer = extract_answer_letter(answer)
        print(answer, final_answer)
        
        results.append(final_answer)    

    print('✅ Inference Done.')

    # submission용 CSV 만들기
    submission = pd.read_csv('eg/sample_submission.csv')
    submission['answer'] = results
    submission.to_csv('results/submission.csv', index=False, encoding='utf-8')
    print("✅ CSV for submission Done.")
    
    # 모델 정보
    kosmos_params = sum(p.numel() for p in kosmos.parameters())
    flan_t5_params = sum(p.numel() for p in flan_t5.parameters())
    print(f"모델 정보:")
    print(f"총 파라미터 수: {kosmos_params + flan_t5_params:,}")


if __name__ == "__main__":
    main()
