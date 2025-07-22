import os
import re
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch

from transformers import AutoProcessor, Kosmos2ForConditionalGeneration, T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel


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


def extract_text_after_prefix(full_text, prefix="Describe this image in detail."):
    """
    주어진 텍스트에서 특정 접두사 뒤에 오는 모든 텍스트를 추출합니다.
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
    first_parsed_text = extract_text_after_prefix(full_text, "Describe this image in detail.")
    
    if first_parsed_text is None:
        return None # 첫 번째 접두사가 없으면 바로 None 반환

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
    
    # flan-t5-large 로드
    t5_directory = os.path.join("Model/flan-t5/final_best_model")
    flan_t5 = T5ForConditionalGeneration.from_pretrained(t5_directory, device_map="auto")   # 모델 로드
    tokenizer = T5Tokenizer.from_pretrained(t5_directory)
    print("✅ flan-t5-large loaded")
    
    # KOSMOS2 로드
    kosmos_directory = os.path.join("Model/kosmos2/2nd_phase/final_best_model")
    kosmos = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224", torch_dtype=torch.bfloat16, device_map="auto" )   # 베이스모델 로드
    kosmos = PeftModel.from_pretrained(kosmos, kosmos_directory, torch_dtype=torch.bfloat16, device_map="auto")        # LoRA weight 적용된 모델 로드
    processor = AutoProcessor.from_pretrained(kosmos_directory)
    print("✅ kosmos2 loaded")
    
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
            output = kosmos.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                image_embeds=None,
                use_cache=True,
                do_sample=True,
                top_k=20,
                top_p=0.9,
                temperature=0.5,
                min_new_tokens=25,      
                max_new_tokens=1024,
                # num_beams
                )
        
        generated_text = processor.batch_decode(output, skip_special_tokens=True)[0]
        description, entities = processor.post_process_generation(generated_text)
        description = extract_description_content(description)
        print(f"Generated {_}:\n {description}\n {row['Question']}\n {choices}")

        # flan-t5-large로 최종 답안 도출
        shot = """
        Question: What might have happened earlier before this scene?
        Choices:
        A. They had a meal in a city restaurant far away
        B. The people gathered their horses from the pastures nearby
        C. They went swimming in a lake after a hot day
        D. They played a horse racing game in the open field
        Description: The image depicts a group of people riding horses on a grassy field. The horses are of various colors and sizes, with some individuals riding in the foreground and others in the background. The sky is partly cloudy, and the overall atmosphere is serene and peaceful.
        
        Before riding horses, especially in a natural setting like a grassy field, the horses would need to be assembled and prepared. Gathering them from nearby pastures is a logical first step for a group outing or activity involving horses.
        The answer is: (B)
        
        """
        prompt = f'Read the description and answer the question by choosing the right one among A, B, C, and D by reasoning step-by-step.\n {shot} Question: {row["Question"]}\n Choices: {choices}\n Description: {description}'
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = flan_t5.generate(
                **inputs,
                use_cache=True,
                do_sample=True,
                top_k=20,
                top_p=0.8,
                temperature=0.5,
                num_return_sequences=5,
                max_new_tokens=2048   
                )
        
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append(extract_answer_letter(answer))
        print('\n'.join([answer, extract_answer_letter(answer)]))


    print('✅ Inference Done.')

    # submission용 CSV 만들기
    submission = pd.read_csv('eg/sample_submission.csv')
    submission['answer'] = results
    submission.to_csv('results/baseline_submit.csv', index=False)
    print("✅ CSV for submission Done.")
    
    # 모델 정보
    kosmos_params = sum(p.numel() for p in kosmos.parameters())
    flan_t5_params = sum(p.numel() for p in flan_t5.parameters())
    print(f"모델 정보:")
    print(f"총 파라미터 수: {kosmos_params + flan_t5_params:,}")


if __name__ == "__main__":
    main()
