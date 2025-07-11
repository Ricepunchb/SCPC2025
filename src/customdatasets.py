import torch
from PIL import Image
from datasets import Dataset
import numpy as np
import re

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
    """A-OKVQA dataset."""

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
    processed = processed.class_encode_column('answer_idx').train_test_split(test_size=0.2, stratify_by_column='answer_idx')
    return processed['train'], processed['test']


# Visual 7w 데이터셋
class VisualDataset(torch.utils.data.Dataset):
    """Visual 7w dataset class"""

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
    """VMC Dataset."""

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
    """Realworld QA Dataset. xai-org/RealworldQA"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        return {"image": sample['image'], 
                "question": sample['question'], 
                "choices": False, 
                "description": sample['description'],
                "answer": chr(65 + sample['answer'])  } 
    
    
    
class DataCollator:
    def __init__(self, processor):
        self.processor = processor

        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id

    def __call__(self, examples):
        '''
        Dataset에서 
        image는 PIL 이미지이길 기대
        choices는 List[str] 형태이길 기대
        description은 하나의 긴 str이길 기대
        answer은 A, B, C, D 중 하나일 것으로 기대        
        '''
        images = []
        texts = []
        instruction_lengths = [] 

        for example in examples:
            images.append(example['image'])
            if example['choices']:
                choices = '\n'.join( f"{chr(65+i)}. {choice}" for i, choice in enumerate(example['choices']) )
                instruction = f"Describe the image in detail. Then choose the correct option from choices.\n Question: {example['question']}\n Choices: {choices}\nAnswer:"
            else:   # This is for Realworld QA only.
                instruction = f"Describe the image in detail. Then choose the correct option from choices.\n Question: {example['question']}\nAnswer:"
            full_answer = f" Description: {example['description']}.\nBy the description and the image given, the final answer is: ({example['answer']})."
            # 최종적으로 모델에 들어갈 전체 텍스트
            texts.append('\n'.join([instruction, full_answer]))
            
            instruction_batch_item = self.processor(
                images=[example['image']], 
                text=[instruction],
                padding=False,
                truncation=False,
                return_tensors="pt"
            )
            # `input_ids`의 길이를 가져옵니다. (여기에는 이미지 토큰도 포함된 길이)
            instruction_lengths.append(instruction_batch_item.input_ids.shape[1])

        batch = self.processor(images=images, text=texts, padding=True, truncation=True, return_tensors="pt")
        
        # `labels`는 `input_ids`를 복사하여 시작합니다.
        labels = batch["input_ids"].clone()
        
        # 안전성 체크
        if labels is None or labels.numel() == 0:
            print(f"ERROR: DataCollator: labels is None or empty. Example inputs that led to this: {examples}")
            raise ValueError("DataCollator produced None or empty labels. Check your data or processor.")
        
        # 패딩 토큰 마스킹: 패딩된 부분은 Loss 계산에서 제외합니다.
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # 각 샘플의 instruction_lengths에 따라 labels의 해당 부분을 -100으로 설정합니다.
        for i, length in enumerate(instruction_lengths):
            # 마스킹할 길이는 실제 `labels` 시퀀스 길이와 `instruction_length` 중 작은 값이어야 합니다.
            mask_len = min(length, labels.shape[1])
            labels[i, :mask_len] = -100 # instruction에 해당하는 토큰들을 -100으로 마스킹

        # 모든 마스킹 작업이 완료된 후, `batch` 딕셔너리에 `labels`를 최종 할당합니다.
        batch["labels"] = labels
        
        return batch
