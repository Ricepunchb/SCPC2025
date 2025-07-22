import os
import re
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import gc
from itertools import chain
import wandb
from evaluate import load
meteor = load("meteor")

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler 
from transformers import AutoProcessor, AutoModelForVision2Seq, Kosmos2ForConditionalGeneration, EarlyStoppingCallback, TrainingArguments, Trainer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, load_from_disk, ClassLabel
from peft import LoraConfig, PeftModel, PeftConfig, get_peft_model

from ..customdatasets import *

        
def compute_meteor(output, tokenizer):
    # 원본 output.predictions는 datalength / batch 의 길이를 가진 tuple. 0번째에 logit들 포함
    # 원본 output.label_ids는 batch size만큼의 길이를 가진 encoded list
    pred_texts = []
    for batch_preds in output.predictions:
        logits = batch_preds[0]
        token_ids = np.argmax(logits, axis=-1)
        decoded_preds = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        pred_texts.extend(decoded_preds)

    label_texts = []
    for sample in output.label_ids:     
        encoded_labels = np.where(sample == -100, tokenizer.pad_token_id, sample)
        decoded_labels = tokenizer.batch_decode(encoded_labels, skip_special_tokens=True) # batch size의 decoded text list가 됨
        label_texts.extend(decoded_labels)

    assert len(pred_texts) == len(label_texts), "Prediction and label counts must match"
    score = meteor.compute(predictions=pred_texts, references=label_texts)
    # print(score)    # DEBUG use
    return {'meteor': float(score['meteor'])}    # {'meteor': np.float64(0.49980072118362123)} 이렇게 생긴 dict임


class WeightedSFTTrainer(SFTTrainer):
    def __init__(self, train_weights, eval_weights, *args, **kwargs):
        super().__init__(*args, **kwargs) # 부모 클래스의 __init__ 호출
        self.train_weights = train_weights # WeightedSFTTrainer에만 필요한 weights 추가
        self.eval_weights = eval_weights
        
    def get_train_dataloader(self) -> DataLoader:
        # 데이터셋 준비
        train_dataset = self.train_dataset
        
        # 1단계에서 계산한 가중치를 사용해 샘플러 생성
        sampler = WeightedRandomSampler(
            weights=self.train_weights,
            num_samples=len(train_dataset),
            replacement=True
        )

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def get_eval_dataloader(self, eval_dataset = None) -> DataLoader:
        sampler = WeightedRandomSampler(
            weights=self.eval_weights,
            num_samples=len(eval_dataset),
            replacement=True
        )

        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory
        )




def main():
    # # !! Important HOTFIX !!
    # import transformers.models.kosmos2.modeling_kosmos2 as kosmos2_module

    # original_forward_embedding = kosmos2_module.Kosmos2TextTransformer.forward_embedding

    # def patched_forward_embedding(self,
    #     input_ids,
    #     inputs_embeds,
    #     image_embeds,
    #     img_input_mask,
    #     past_key_values_length,
    #     position_ids):
    #     if inputs_embeds is not None:
    #         inputs_embeds = inputs_embeds.clone()
    #     return original_forward_embedding(
    #         self, input_ids, inputs_embeds, image_embeds,
    #         img_input_mask, past_key_values_length, position_ids
    #     )
    # kosmos2_module.Kosmos2TextTransformer.forward_embedding = patched_forward_embedding

    # WandB 시동
    wandb.init(project="SCPC 2025")

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

    # --------------------- Dataset 선언 -------------------------------
    # Realworld QA 데이터셋 로드
    ds = load_from_disk("datasets/Realworld/Realworld_aug")
    ds = ds.filter(lambda x: x['answer'] in ['A', 'B', 'C', 'D'])
    ds = ds.class_encode_column("answer").train_test_split(test_size=0.2, stratify_by_column='answer')
    train_ds_1 = RealworldDataset(ds['train'])
    val_ds_1 = RealworldDataset(ds['test'])
    
    train_ds_2 = load_dataset("HuggingFaceM4/A-OKVQA", split="train")
    val_ds_2 = load_dataset("HuggingFaceM4/A-OKVQA", split="validation")
    val_ds_2 = val_ds_2.shuffle().select(range(300))
    train_ds_2 = AokvqaDataset(train_ds_2)
    val_ds_2 = AokvqaDataset(val_ds_2)
    
    ds = load_dataset("csv", data_files="datasets/stanford_img_para_caption/stanford_df_rectified.csv", split='train')
    train_ds_3 = ds.filter(lambda x: x['train'])
    val_ds_3 = ds.filter(lambda x: x['test'])
    val_ds_3 = val_ds_3.shuffle().select(range(300))
    train_ds_3 = StafordDataset(train_ds_3)
    val_ds_3 = StafordDataset(val_ds_3)
    
    train_ds_4 = load_dataset("json", data_files="eg/train_aug.jsonl", split='train')
    train_ds_4 = train_ds_4.train_test_split(test_size=0.2)
    val_ds_4 = DACONDataset(train_ds_4['test'])
    train_ds_4 = DACONDataset(train_ds_4['train'])
    
    train_ds = ConcatDataset([train_ds_1, train_ds_2, train_ds_3, train_ds_4])
    val_ds = ConcatDataset([val_ds_1, val_ds_2, val_ds_3, val_ds_4])
    print(f"Total number of train samples: {len(train_ds)}")
    print(f"Total number of validation samples: {len(val_ds)}")
    train_weights = [[1.0/len(d)] * len(d) for d in [train_ds_1, train_ds_2, train_ds_3, train_ds_4] ]
    train_weights = list(chain.from_iterable(train_weights))
    assert len(train_weights) == len(train_ds)
    
    val_weights = [[1.0/len(d)] * len(d) for d in [val_ds_1, val_ds_2, val_ds_3, val_ds_4] ]
    val_weights = list(chain.from_iterable(val_weights))
    
    # 메모리 정리
    del ds, train_ds_1, train_ds_2, train_ds_3, train_ds_4, val_ds_1, val_ds_2, val_ds_3, val_ds_4
    torch.cuda.empty_cache() # GPU 캐시 비우기
    gc.collect()             # Python 가비지 컬렉터 실행
    
    # # --------------------- 모델 선언 -----------------------
    model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224", device_map='auto', torch_dtype=torch.bfloat16, _attn_implementation="sdpa",)
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224", add_eos_token=True)    
    # --------------------- [Optional] 모델 로드 for continual training -----------------------
    # final_save_directory = os.path.join("Model/kosmos2/1st_phase/final_best_model")
    # model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224", torch_dtype=torch.bfloat16, device_map="auto" )   # 베이스모델 로드
    # model = PeftModel.from_pretrained(model, final_save_directory, torch_dtype=torch.bfloat16, device_map="auto", is_trainable=True )        # LoRA weight 적용된 모델 로드
    # processor = AutoProcessor.from_pretrained(final_save_directory)
    
    # ------------------------------ LoRA Config -----------------------------------
    target_modules=[
        # Text-Transformer 내의 Linear들 (substring match)
        "k_proj", "v_proj", "q_proj", "out_proj",
        "fc1", "fc2",
        # image→text projection 의 dense 레이어
        "image_to_text_projection.dense",
    ]
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules = target_modules,
        modules_to_save=["lm_head", "embed_tokens"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    
    # 모델 정보
    print(f"모델 정보:")
    # total_params = sum(p.numel() for p in model.parameters())
    # print(total_params)
    model.print_trainable_parameters()
    
    # --------------------- SFTTrainer 설정 -----------------------
    data_collator = DataCollator(processor, model_dtype=torch.bfloat16)
    
    batch_size = 16

    training_args = SFTConfig(
        report_to='wandb',
        run_name = 'run-0721',
        output_dir="Model/kosmos2/1st_phase",
        overwrite_output_dir=True,
        num_train_epochs=3,     # epochs
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=4,
        eval_do_concat_batches=False,
        eval_accumulation_steps=2,
        gradient_accumulation_steps=128 // batch_size,
        warmup_ratio=0.05,
        learning_rate=1e-4,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        lr_scheduler_kwargs = {"num_cycles": 2.5},
        logging_steps=10,
        eval_strategy='steps',
        eval_steps=0.1,
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        greater_is_better=True,
        load_best_model_at_end=True,
        optim="adamw_torch_fused",
        bf16=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataset_kwargs={"skip_prepare_dataset": True},
        label_names=["labels"],
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        dataloader_pin_memory=True
        )
    
    trainer = WeightedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics= lambda eval_pred: compute_meteor(eval_pred, processor.tokenizer),
        train_weights=train_weights,
        eval_weights=val_weights
        # callbacks=[early_stop_callback]
    )
    
    print("✅ 1st Training start.")
    trainer.train()
    print("✅ 1st Training done.")
    
    # 모델 수동 저장
    final_save_directory = os.path.join(training_args.output_dir, "final_best_model")
    os.makedirs(final_save_directory, exist_ok=True)
    trainer.model.save_pretrained(final_save_directory)     # PEFT 어댑터 (LoRA 가중치)와 PEFT 설정만 저장
    print(f"PEFT adapter saved to {final_save_directory}")
    processor.save_pretrained(final_save_directory)     # processor 저장
    print(f"Processor saved to {final_save_directory}")
    
    # 베스트 모델 로드
    model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224", torch_dtype=torch.bfloat16, device_map="auto" )   # 베이스모델 로드
    model = PeftModel.from_pretrained(model, final_save_directory, torch_dtype=torch.bfloat16, device_map="auto", is_trainable=True )        # LoRA weight 적용된 모델 로드
    processor = AutoProcessor.from_pretrained(final_save_directory)
    
    # ---------------------------------------------------------  Recap-COCO-30K 데이터셋 훈련 ---------------------------------------------------------
    ds = load_dataset("UCSC-VLAA/Recap-COCO-30K", split="train")
    ds = ds.train_test_split(test_size=700)
    train_ds = RecapCOCODataset(ds['train'])
    val_ds = RecapCOCODataset(ds['test'])
    
    # 메모리 정리
    del ds, trainer, training_args
    torch.cuda.empty_cache() # GPU 캐시 비우기
    gc.collect()             # Python 가비지 컬렉터 실행
    
    training_args = SFTConfig(
        report_to='wandb',
        run_name = 'run-0721',
        output_dir="Model/kosmos2/2nd_phase",
        overwrite_output_dir=True,
        num_train_epochs=3,     # epochs
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=4,
        eval_do_concat_batches=False,
        eval_accumulation_steps=2,
        gradient_accumulation_steps=128 // batch_size,
        warmup_ratio=0.05,
        learning_rate=5e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        lr_scheduler_kwargs = {"num_cycles": 2.5},
        logging_steps=10,
        eval_strategy='steps',
        eval_steps=0.1,
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=1,
        metric_for_best_model="meteor",
        greater_is_better=True,
        load_best_model_at_end=True,
        optim="adamw_torch_fused",
        bf16=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataset_kwargs={"skip_prepare_dataset": True},
        label_names=["labels"],
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        dataloader_pin_memory=True
        )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics= lambda eval_pred: compute_meteor(eval_pred, processor.tokenizer),
        # callbacks=[early_stop_callback]
    )
    
    print("✅ 2nd Training start.")
    trainer.train()
    print("✅ 2nd Training done.")
    
    # 모델 수동 저장
    final_save_directory = os.path.join(training_args.output_dir, "final_best_model")
    os.makedirs(final_save_directory, exist_ok=True)
    trainer.model.save_pretrained(final_save_directory)     # PEFT 어댑터 (LoRA 가중치)와 PEFT 설정만 저장
    print(f"PEFT adapter saved to {final_save_directory}")
    processor.save_pretrained(final_save_directory)     # processor 저장
    print(f"Processor saved to {final_save_directory}")
    
    # 베스트 모델 로드
    model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224", torch_dtype=torch.bfloat16, device_map="auto" )   # 베이스모델 로드
    model = PeftModel.from_pretrained(model, final_save_directory, torch_dtype=torch.bfloat16, device_map="auto", is_trainable=True )        # LoRA weight 적용된 모델 로드
    processor = AutoProcessor.from_pretrained(final_save_directory)
    
    # ----------------------------------------------- 트레이닝 결과 확인 w/ Test dataset --------------------------------------------------
    # 메모리 정리
    del train_ds, val_ds, trainer
    torch.cuda.empty_cache() # GPU 캐시 비우기
    gc.collect()             # Python 가비지 컬렉터 실행
    
    test = pd.read_csv('eg/test.csv')[:10]
    
    model.eval()
    
    for _, row in tqdm(test.iterrows(), total=len(test)):
        image = Image.open('eg/'+row['img_path']).convert("RGB")
        choices = '\n'.join([f"{c}. {row[c]}" for c in ['A', 'B', 'C', 'D']])
        prompt = f"Describe this image in detail."

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                image_embeds=None,
                use_cache=True,
                do_sample=False,
                # top_k=5,
                # top_p=0.9,
                # temperature=0.5,
                min_new_tokens=128,      
                max_new_tokens=2048,
                num_beams=3
                )
        generated_text = processor.batch_decode(output, skip_special_tokens=True)[0]
        answer, entities = processor.post_process_generation(generated_text)
        
        print("Generated: ", answer)

    print('✅ Inference Done.')


if __name__ == "__main__":
    main()