import torch
from PIL import Image
from datasets import Dataset, load_dataset
import numpy as np
import re
import random

# DACON 오피셜 데이터셋
class DACONDataset(torch.utils.data.Dataset):
    """DACON official dataset."""

    def __init__(self, dataset):
        self.dataset = dataset.remove_columns(["ID"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get image + text
        sample = self.dataset[idx]
        
        return {"image": Image.open(f"eg/{sample['img_path']}").convert('RGB'), 
                "question": sample['Question'], 
                "choices": [sample[c] for c in ['A', 'B', 'C', 'D']], 
                "description": sample['description'],
                "answer": sample['answer']  }
        
    
    
# A-OKVQA 데이터셋
class AokvqaDataset(torch.utils.data.Dataset):
    """
    A-OKVQA dataset.
    https://github.com/allenai/aokvqa
    Apache-2.0 license
        상업적 이용과 수정, 재배포가 모두 허용
        원저작자(저작권자) 정보 명시
        라이선스 전문 포함 (LICENSE 파일 같이 배포)
    """

    def __init__(self, dataset):
        self.dataset = dataset.remove_columns(["question_id", 'direct_answers', 'difficult_direct_answer'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get image + text
        sample = self.dataset[idx]
        
        return {"image": sample['image'].convert('RGB'), 
                "question": sample['question'], 
                "choices": sample['choices'], 
                "description": ' '.join(sample['rationales']),
                "answer": chr(65 + sample['correct_choice_idx']) } 
    
    
# Visual 7w 데이터셋 전처리
def preproc_visual7w(dataset):
    '''
    visual7w dataset preprocess and stratified split into train test datasets
    '''
    processed = []
    for sample in dataset:
        image_name = sample['filename']

        for qa in sample['qa_pairs']:
            answer_idx = np.random.randint(0,4)
            choices = qa['multiple_choices']
            choices.insert(answer_idx, qa['answer'])
            processed.append({'question': qa['question'],
                              'answer': ['A', 'B', 'C', 'D'][answer_idx],
                              'answer_idx': answer_idx,
                              'choices': choices,
                              'img_path': f'datasets/visual7w/images/{image_name}',
                              'description': sample['description']  })
    processed = Dataset.from_list(processed)
    processed = processed.class_encode_column('answer_idx').train_test_split(test_size=100, stratify_by_column='answer_idx')
    return processed['train'], processed['test']


# Visual 7w 데이터셋
class VisualDataset(torch.utils.data.Dataset):
    """
    Visual 7w dataset class
    https://github.com/yukezhu/visual7w-toolkit
    License: MIT 아마도?
        상업적, 비상업적 모두 자유롭게 사용, 수정, 배포 가능합니다.
        소프트웨어를 사용하거나 배포할 때 저작권 고지 및 원본 MIT 라이선스 문구를 포함하는 것만 의무
    Visual Genome	CC BY 4.0
    COCO	CC BY 4.0 (일반적으로)
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        return {"image": Image.open(sample['img_path']).convert('RGB'), 
                "question": sample['question'], 
                "choices": sample['choices'], 
                "description": sample['description'],
                "answer": sample['answer']  }
    
    
# VMCDataset
class VMCDataset(torch.utils.data.Dataset):
    """
    VMC Dataset.
    https://huggingface.co/datasets/suyc21/VMCBench
    License: MIT
    """

    def __init__(self, dataset):
        self.dataset = dataset.remove_columns(["category"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get image + text
        sample = self.dataset[idx]

        return {"image": sample['image'].convert('RGB'), 
                "question": sample['question'], 
                "choices": [sample[c] for c in ['A', 'B', 'C', 'D']], 
                "description": sample['description'],
                "answer": chr(65 + sample['answer'])  }
        
        
# RealworldDataset
class RealworldDataset(torch.utils.data.Dataset):
    """
    Realworld QA Dataset.
    https://huggingface.co/datasets/xai-org/RealworldQA
    License: CC BY-ND 4.0.
        상업적 이용이 가능합니다.
        저작자 및 출처 표기가 반드시 필요합니다.
        변경 또는 2차 저작물 제작이 금지됩니다. 즉, 원본 그대로만 복사, 배포, 이용이 가능
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        return {"image": sample['image'], 
                "question": sample['question'], 
                "choices": [], 
                "description": sample['description'],
                "answer": chr(65 + sample['answer'])  } 
        
        
# Stanford_img_paragraph_captioning dataset
class StanfordDataset(torch.utils.data.Dataset):
    """
    Stanford_img_paragraph_captioning Dataset.
    https://www.kaggle.com/datasets/vakadanaveen/stanford-image-paragraph-captioning-dataset?select=stanford_df_rectified.csv
    License: CC0: Public Domain
        상업적/비상업적 구분 없이 누구나 자유롭게 저작물을 복사, 수정, 배포, 재사용, 2차 창작, 심지어 판매 등 모든 목적으로 사용할 수 있습니다.
        저작자 표시 의무도 없습니다
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        return {
            "image": Image.open(f"datasets/stanford_img_para_caption/stanford_img/content/stanford_images/{sample['Image_name']}.jpg").convert('RGB'),
            "question": "",
            "choices": [], 
            "description": sample['Paragraph'],
            "answer": ""
            }
        

# RecapCOCODataset
class RecapCOCODataset(torch.utils.data.Dataset):
    """
    UCSC-VLAA/Recap-COCO-30K
    https://huggingface.co/datasets/UCSC-VLAA/Recap-COCO-30K
    License: cc-by-4.0
        반드시 저작자 표시(출처)를 해야 합니다.
    """

    def __init__(self, dataset):
        self.dataset = dataset.remove_columns(["image_id", "coco_url", "caption"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get image + text
        sample = self.dataset[idx]

        return {"image": sample['image'].convert('RGB'), 
                "question": "", 
                "choices": [], 
                "description": sample['recaption'],
                "answer": ""  }


class DataCollator_1:
    def __init__(self, processor, model_dtype=torch.float32):
        self.processor = processor
        self.model_dtype = model_dtype

    def __call__(self, examples):
        '''
        Dataset에서 
        image는 PIL 이미지이길 기대
        choices는 List[str] 형태이길 기대
        description은 하나의 긴 str이길 기대
        answer은 A, B, C, D 중 하나일 것으로 기대        
        
        Image description 훈련용
        '''
        images = []
        instructions = [] 
        full_text = []
        
        for example in examples:
            images.append(example['image'])
            instruction = f"Describe this image in detail."
            instructions.append(instruction)
            answer = (f"\nDescription: {example['description']}")
            full_text.append('\n'.join([instruction, answer]))

        batch = self.processor(images=images, text=full_text, padding=True, return_tensors="pt")
        instruction_batch = self.processor(images=images, text=instructions, padding=True, return_tensors="pt", add_eos_token=False)
        
        # `labels`는 `input_ids`를 복사하여 시작합니다.
        labels = batch["input_ids"].clone()
        
        # 안전성 체크
        if labels is None or labels.numel() == 0:
            print(f"ERROR: DataCollator: labels is None or empty. Example inputs that led to this: {examples}")
            raise ValueError("DataCollator produced None or empty labels. Check your data or processor.")
        
        # pad token 마스킹: 패딩된 부분은 Loss 계산에서 제외.
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        # if len(examples) < 3:
        #     random_indices = list(range(len(examples)))
        # else:
        #     random_indices = random.sample(range(len(examples)), 3)
            
        # 각 샘플의 instruction_lengths에 따라 labels의 해당 부분을 -100으로 설정합니다.
        for i in range(len(examples)):
            # instruction 배치에서 실제 토큰 길이 계산 (패딩 제외)
            instruction_tokens = instruction_batch["input_ids"][i]
            # 패딩이 아닌 토큰들의 개수 계산
            non_pad_mask = instruction_tokens != self.processor.tokenizer.pad_token_id
            actual_instruction_length = non_pad_mask.sum().item()
            # instruction 부분을 -100으로 마스킹
            mask_len = min(actual_instruction_length, labels.shape[1])
            labels[i, :mask_len] = -100
            
            # # DEBUG
            # if i in random_indices: # 선택된 인덱스일 경우에만 출력
            #     valid_tokens = labels[i][labels[i] != -100]  # -100이 아닌 토큰들만 추출
            #     if len(valid_tokens) > 0:
            #         decoded_text = self.processor.tokenizer.decode(valid_tokens, skip_special_tokens=False)
            #         print(f"DEBUG Sample {i}: \n{decoded_text}")
            #     else:
            #         print(f"DEBUG Sample {i}: No valid tokens to decode (all masked or empty)")

        batch["labels"] = labels
        
        return batch   
    
    
class T5DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        '''
        Dataset에서 
        choices는 List[str] 형태이길 기대
        description은 하나의 긴 str이길 기대
        answer은 A, B, C, D 중 하나일 것으로 기대        
        '''
        instructions = []
        answers = []

        for example in examples:
            if not example.get('choices'):  # for RealWorldQA
                instructions.append(f"Read the context and answer the question by choosing the right option among A, B, C, and D.\n Context: {example['description']} \n Question: {example['question']} ")
                answers.append(f"The answer is: ({example['answer']})")
            else:    # standard QA cases
                formatted_choice = '\n'.join( [f'{chr(65+i)}. {c}' for i, c in enumerate(example['choices']) ] ) 
                instructions.append(f"Read the context and answer the question by choosing the right option among A, B, C, and D.\n Context: {example['description']} \n Question: {example['question']} \n Choices {formatted_choice}")
                answers.append(f"The answer is: ({example['answer']})")

        batch = self.tokenizer(instructions, padding=True, truncation=True, max_length=768, return_tensors="pt")
        labels = self.tokenizer(text_target=answers, padding=True, return_tensors="pt", add_special_tokens=True)
        labels = labels['input_ids']
        
        # 안전성 체크
        if labels is None or len(labels) == 0:
            print(f"ERROR: DataCollator: labels is None or empty. Example inputs that led to this: {examples}")
            print(f"Sample answers: {answers[:3]}")
            raise ValueError("DataCollator produced None or empty labels. Check your data or tokenizer.")
            
        # 패딩 토큰 마스킹: 패딩된 부분은 Loss 계산에서 제외합니다.
        labels[labels == self.tokenizer.pad_token_id] = -100

        # # DEBUG
        # for i in range(len(examples)):
        #     print(f"\n=== DataCollator Debug Info ===")
        #     print(f"3 Sample labels : {labels[:3]}")

        batch["labels"] = labels
        
        return batch