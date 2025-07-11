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
from transformers import AutoProcessor, AutoModelForVision2Seq, Kosmos2ForConditionalGeneration
from trl import SFTTrainer, SFTConfig

from datasets import load_dataset
from peft import LoraConfig, PeftModel, PeftConfig


# 정답 알파벳 추출 함수
def extract_answer_letter(text):
    match = re.search(r"\s*([A-Da-d])\s*", text)
    return match.group(1).upper() if match else "?"


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
    
    # ----------------------------------------------- 제출용 추론 --------------------------------------------------
    # 베스트 모델 로드
    final_save_directory = os.path.join("Models/kosmos2", "final_best_model")
    peft_config = PeftConfig.from_pretrained(final_save_directory)     # 훈련후 best checkpoint
    model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224", torch_dtype=torch.bfloat16, device_map="auto" )   # 베이스모델 로드
    model = PeftModel.from_pretrained(model, final_save_directory, torch_dtype=torch.bfloat16, device_map="auto" )        # LoRA weight 적용된 모델 로드
    processor = AutoProcessor.from_pretrained(final_save_directory)
    
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