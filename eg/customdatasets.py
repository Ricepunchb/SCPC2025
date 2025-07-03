import torch
from PIL import Image


# InstructBLIP은 일반적으로 답을 프롬프트에 포함하여 훈련합니다.
# InstructBLIPProcessor는 이미지를 pixel_values로, 텍스트를 input_ids, attention_mask 등으로 변환합니다.
# IMPORTANT: 'labels'는 모델의 타겟 아웃풋이므로, 이 프롬프트 자체를 인코딩할 때 포함시키지 않습니다.
# labels는 별도로 처리해야 합니다.
# 이 부분에서 `text`에는 질문과 옵션, 그리고 "Answer:"까지 포함된 프롬프트를 전달합니다.
# InstructBLIPForConditionalGeneration의 forward 메서드는 `labels` 인자를 받습니다.
# 이 `labels`는 모델이 생성해야 할 "정답" 부분에 해당합니다.
# 따라서 `processor`를 호출할 때는 `text`에 질문과 옵션만 주고, `labels`를 별도로 생성해야 합니다.
# InstructBLIP은 보통 <image> 토큰을 사용하고, Answer: 뒤에 LLM이 생성할 텍스트를 기대합니다.
# 따라서 `text`에는 Answer: 앞부분까지의 프롬프트를 주고, `labels`에는 Answer: 뒷부분의 정답을 토큰화하여 제공해야 합니다.
# VQA 태스크의 경우, InstructBLIP의 일반적인 훈련 방식은 "Question: <question_text> Answer: <answer_text>"입니다.
# Multiple Choice의 경우, "Question: <question_text> Options: <options_text> Answer: <answer_text>"가 될 수 있습니다.
# `processor`의 `text` 인자에 전체 프롬프트 (정답 포함)를 넣어주고, `labels`는 나중에 콜레이트 함수에서 처리합니다.

# InstructBLIP의 기본적인 VQA Fine-tuning 예시를 참고하면,
# prompt = f"Question: {sample['question']} Answer:"
# answer_text = sample['choices'][sample['correct_choice_idx']]

# VQA with Multiple Choice:
# processor는 `pixel_values`, `qformer_input_ids`, `qformer_attention_mask`, `input_ids`, `attention_mask`를 생성합니다.
# 여기서 `input_ids`와 `attention_mask`는 LLM에 들어갈 전체 프롬프트 (Q-Former 출력 삽입 자리 포함) 입니다.
# 'labels'는 모델이 생성해야 할 '정답' 텍스트를 토큰화한 것입니다.
# InstructBlipProcessor는 텍스트를 인코딩할 때 이미 LLM의 토크나이저를 사용하므로,
# labels도 동일한 토크나이저로 인코딩해야 합니다.

# `labels`는 `generate`에서 사용되는 것이 아니라, `forward` 메서드의 타겟으로 사용됩니다.
# 모델은 `input_ids`를 입력받아 `labels`와 비교하여 손실을 계산합니다.
# 따라서 `labels`는 LLM이 "Answer: " 이후에 생성해야 할 정답 텍스트를 토큰화한 것이어야 합니다.

# AokvqaDataset.__getitem__에서는 `labels`를 정답 텍스트를 인코딩한 형태로 저장합니다.
# 실제 모델 훈련 시에는 이 `labels`가 LLM의 출력과 비교됩니다.

# 커스텀 데이터셋
class AokvqaDataset(torch.utils.data.Dataset):
    """A-OKVQA dataset."""

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get image + text
        sample = self.dataset[idx]
        answer_idx = sample['correct_choice_idx']
        answer = sample['choices'][answer_idx]
        image = sample['image'].convert('RGB')
        prompt = f"""
        <image>
        Based on the image, choose the correct option to the following question.

        Question: {sample['question']}

        Options:
        A {sample['choices'][0]}
        B {sample['choices'][1]}
        C {sample['choices'][2]}
        D {sample['choices'][3]}

        Answer:
        """
        encoding = self.processor(image, prompt, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        label_encoding = self.processor.tokenizer(
            answer,
            max_length=128, # 라벨의 최대 길이 (충분히 크게)
            padding="max_length", # 여기서는 max_length 패딩이 적절
            truncation=True,
            return_tensors="pt"
        )
        encoding["labels"] = label_encoding.input_ids.squeeze(0)
        encoding["labels"][encoding["labels"] == self.processor.tokenizer.pad_token_id] = -100
        return encoding

    
class DaconDataset(torch.utils.data.Dataset):
    """Official dataset from DACON."""

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get image + text
        sample = self.dataset[idx]
        image = Image.open(sample['img_path']).convert("RGB")
        answer = sample[sample['answer'].strip()]
        
        prompt = f"""
        <image>
        Based on the image, choose the correct option to the following question.

        Question: {sample['Question']}

        Options:
        A {sample['A']}
        B {sample['B']}
        C {sample['C']}
        D {sample['D']}

        Answer:
        """
        encoding = self.processor(image, prompt, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        label_encoding = self.processor.tokenizer(
            answer,
            max_length=128, # 라벨의 최대 길이 (충분히 크게)
            padding="max_length", # 여기서는 max_length 패딩이 적절
            truncation=True,
            return_tensors="pt"
        )
        encoding["labels"] = label_encoding.input_ids.squeeze(0)
        encoding["labels"][encoding["labels"] == self.processor.tokenizer.pad_token_id] = -100
        return encoding
    
    
def collate_fn(batch):
    # 'pixel_values', 'qformer_input_ids', 'qformer_attention_mask', 'input_ids', 'attention_mask', 'labels'
    # 이 모든 키들이 배치로 쌓여야 합니다.
    processed_batch = {}
    for key in batch[0].keys():
        processed_batch[key] = torch.stack([sample[key] for sample in batch])
        
    return processed_batch