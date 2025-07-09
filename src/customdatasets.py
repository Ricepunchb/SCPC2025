import torch
from PIL import Image
from datasets import Dataset
import numpy as np

# DACON 오피셜 데이터셋
class DACONDataset(torch.utils.data.Dataset):
    """DACON official dataset."""

    def __init__(self, dataset, processor):
        self.dataset = dataset.remove_columns(["ID"])
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get image + text
        sample = self.dataset[idx]
        
        image = Image.open(f"eg/{sample['img_path']}").convert('RGB')
        question = sample['Question']
        choices = [sample[c] for c in ['A', 'B', 'C', 'D']]
        answer = sample['answer']
        
        
        return {"image": image, "question": question, "choices": choices, "answer": answer}
    
    
# A-OKVQA 데이터셋
class AokvqaDataset(torch.utils.data.Dataset):
    """A-OKVQA dataset."""

    def __init__(self, dataset, processor):
        self.dataset = dataset.remove_columns(["question_id", 'direct_answers', 'difficult_direct_answer', 'rationales'])
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get image + text
        sample = self.dataset[idx]
        
        question = sample['question']
        choices = sample['choices']
        answer = list(zip(choices, ['A', 'B', 'C', 'D']))[sample['correct_choice_idx']][1]
        image = sample['image'].convert('RGB')
        
        return {"image": image, "question": question, "choices": choices, "answer": answer}    
    
    
# Visual 7w 데이터셋 전처리
def preproc_visual7w(dataset):
    processed = []
    for sample in dataset:
        sample = sample['images']
        image_name = sample['filename']

        for i in range(len(sample['qa_pairs'])):
            answer_idx = np.random.randint(0,4)
            choices = sample['qa_pairs'][i]['multiple_choices']
            choices.insert(answer_idx, sample['qa_pairs'][i]['answer'])
            processed.append({'question': sample['qa_pairs'][i]['question'],
                              'answer': ['A', 'B', 'C', 'D'][answer_idx],
                              'answer_idx': answer_idx,
                              'choices': choices,
                              'img_path': f'datasets/visual7w/images/{image_name}'   })
    processed = Dataset.from_list(processed)
    processed = processed.class_encode_column('answer_idx').train_test_split(test_size=0.2, stratify_by_column='answer_idx')
    return processed['train'], processed['test']


# Visual 7w 데이터셋
class VisualDataset(torch.utils.data.Dataset):
    """Visual 7w dataset."""

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get image + text
        sample = self.dataset[idx]
        
        question = sample['question']
        choices = sample['choices']
        answer = sample['answer']
        image = Image.open(sample['img_path']).convert('RGB')
        
        return {"image": image, "question": question, "choices": choices, "answer": answer}
    
    
    
class DataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        prompt_lens = []

        for example in examples:
            image = example['image']
            question = example["question"]
            answer = example['answer']
            choices = '\n'.join([f"{chr(65+i)}. {c}" for i, c in enumerate(example['choices'])])

            # prompt 텍스트 생성
            prompt = f"Question: {question}\nChoices: {choices}\nAnswer:"
            full_text = f"{prompt} {answer}"
            texts.append(full_text)
            images.append(image)

            # prompt 길이를 나중에 마스킹에 쓰기 위해 저장
            prompt_input_ids = self.processor.tokenizer(prompt, return_tensors="pt").input_ids[0]
            prompt_lens.append(len(prompt_input_ids))

        # 전체 text + image 처리
        batch = self.processor(images=images, text=texts, padding='longest', max_length=2048, truncation=True, return_tensors="pt")
        input_ids = batch["input_ids"]

        # labels는 input_ids 복사
        labels = input_ids.clone()

        # 각 sample마다 prompt 길이만큼 -100으로 마스킹
        for i, prompt_len in enumerate(prompt_lens):
            labels[i, :prompt_len] = -100

        # pad 토큰도 무시하도록 마스킹
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        batch["labels"] = labels
        return batch
