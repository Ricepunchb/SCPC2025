import torch
from PIL import Image
from datasets import Dataset, load_dataset
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
    processed = processed.class_encode_column('answer_idx').train_test_split(test_size=0.1, stratify_by_column='answer_idx')
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
                "choices": [], 
                "description": sample['description'],
                "answer": chr(65 + sample['answer'])  } 
        
        
# Staford_img_paragraph_captioning dataset
class StafordDataset(torch.utils.data.Dataset):
    """
    Staford_img_paragraph_captioning Dataset.
    https://www.kaggle.com/datasets/vakadanaveen/stanford-image-paragraph-captioning-dataset?select=stanford_df_rectified.csv
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
        

# Explanations for CommonsenseQA
class ECQADataset(torch.utils.data.Dataset):
    """
    Explanations for CommonsenseQA
    https://github.com/IBM/ecqa
    """

    def __init__(self, dataset):
        ds = load_dataset("json", data_files="/mnt/workspace/datasets/ecqa/ecqa.jsonl", split="train")
        ds_mapping = {example['id']: example['explanation'] for example in ds}
        del ds
        def add_explanation(example):
            current_id = example['id']
            example['explanation'] = ds_mapping.get(current_id, None)
            return example
        
        self.dataset = dataset.remove_columns(["question_concept"])
        self.dataset = self.dataset.map(add_explanation)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]

        return {"image": None, 
                "question": row['question'], 
                "choices": row['choices']['text'],
                "answer": row['answerKey'],
                "explanation": row['explanation'],}
    
    
class DataCollator:
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
        '''
        images = []
        texts = []
        instructions = [] 

        for example in examples:
            images.append(example['image'])
            instruction = f"Describe this image in detail."
            instructions.append(instruction)
            
            full_answer = f"Description: {example['description']}"
            texts.append('\n'.join([instruction, full_answer]))

        batch = self.processor(images=images, text=texts, padding=True, return_tensors="pt", max_length=1024)
        instruction_batch = self.processor(images=images, text=instructions, padding=True, return_tensors="pt", max_length=1024, add_eos_token=False)
        
        # `labels`는 `input_ids`를 복사하여 시작합니다.
        labels = batch["input_ids"].clone()
        
        # 안전성 체크
        if labels is None or labels.numel() == 0:
            print(f"ERROR: DataCollator: labels is None or empty. Example inputs that led to this: {examples}")
            raise ValueError("DataCollator produced None or empty labels. Check your data or processor.")
        
        # pad token 마스킹: 패딩된 부분은 Loss 계산에서 제외.
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
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
            
            # # DEBUG:: -100으로 마스킹되지 않은 부분만 디코딩해서 출력
            # valid_tokens = labels[i][labels[i] != -100]  # -100이 아닌 토큰들만 추출
            # if len(valid_tokens) > 0:
            #     decoded_text = self.processor.tokenizer.decode(valid_tokens, skip_special_tokens=False)
            #     print(f"Sample {i} - Valid labels decoded: \n{decoded_text}")
            # else:
            #     print(f"Sample {i} - No valid tokens to decode")

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
            elif not example.get('explanation'):    # standard QA cases
                formatted_choice = '\n'.join( [f'{chr(65+i)}. {c}' for i, c in enumerate(example['choices']) ] ) 
                instructions.append(f"Read the context and answer the question by choosing the right option among A, B, C, and D.\n Context: {example['description']} \n Question: {example['question']} \n Choices {formatted_choice}")
                answers.append(f"The answer is: ({example['answer']})")
            else:   # for ECQA
                formatted_choice = '\n'.join( [f'{chr(65+i)}. {c}' for i, c in enumerate(example['choices']) ] ) 
                instructions.append(f"Answer the question by choosing the right option among A, B, C, and D by reasoning.\n Question: {example['question']}\n Choices: {formatted_choice}")
                answers.append(f"{example['explanation']} \nThe answer is: ({example['answer']})")

        batch = self.tokenizer(instructions, padding=True, return_tensors="pt")
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
