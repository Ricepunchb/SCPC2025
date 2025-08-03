import os
import re
import random
import warnings
import numpy as np
import gc
from itertools import chain

from evaluate import load
meteor = load("meteor")

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler 
from transformers import AutoProcessor,  Kosmos2ForConditionalGeneration, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, load_from_disk, ClassLabel
from peft import LoraConfig, PeftModel, get_peft_model

from ..customdatasets import *


class WeightedSFTTrainer(SFTTrainer):
    def __init__(self, train_weights=None, eval_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs) # 부모 클래스의 __init__ 호출
        self.train_weights = train_weights # WeightedSFTTrainer에만 필요한 weights 추가
        self.eval_weights = eval_weights
        
    def get_train_dataloader(self) -> DataLoader:
        # 데이터셋 준비
        train_dataset = self.train_dataset
        
        # Weigthed Random Sampler 생성
        if not self.train_weights:
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
            
        else:
            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        
    
    def get_eval_dataloader(self, eval_dataset = None) -> DataLoader:
        if not self.eval_weights:
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
            
        else:
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory
            )
            

def compute_meteor(output, tokenizer):
    print("--- compute_accuracy 함수 시작 ---")
    logits = output.predictions[0]
    predicted_token_ids = np.argmax(logits, axis=-1)
    pred_texts = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)

    label_ids_processed = np.where(output.label_ids == -100, tokenizer.pad_token_id, output.label_ids)
    label_texts = tokenizer.batch_decode(label_ids_processed, skip_special_tokens=True)
    
    # DEBUG
    print(f"\n디코딩된 예측 텍스트 샘플 (첫 1개):\n{pred_texts[:3]}")
    print(f"\n디코딩된 레이블 텍스트 샘플 (첫 1개):\n{label_texts[:3]}")

    assert len(pred_texts) == len(label_texts), "Prediction and label counts must match"

    # METEOR 계산
    result = meteor.compute(predictions=pred_texts, references=label_texts)
    print("--- compute_accuracy 함수 종료 ---")

    return result
    

def main():
    # !! Important HOTFIX !!
    # 기본 모듈 포워딩에 inplace 연산 에러가 있어서 핫픽스.
    import transformers.models.kosmos2.modeling_kosmos2 as kosmos2_module
    import torch.nn as nn
    from typing import Optional

    def patched_forward_embedding(
            self,
            input_ids,
            inputs_embeds: Optional[torch.Tensor] = None,
            image_embeds: Optional[torch.Tensor] = None,
            img_input_mask: Optional[torch.Tensor] = None,
            past_key_values_length: int = 0,
            position_ids: Optional[torch.Tensor] = None,
        ):
        # The argument `inputs_embeds` should be the one without being multiplied by `self.embed_scale`.
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if image_embeds is not None:
            inputs_embeds = inputs_embeds.clone()   ##### HOTFIX added this line
            inputs_embeds[img_input_mask.to(dtype=torch.bool)] = image_embeds.to(inputs_embeds.device).view(
                -1, image_embeds.size(-1)
            )

        inputs_embeds = inputs_embeds * self.embed_scale

        # embed positions
        positions = self.embed_positions(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            position_ids=position_ids,
        )
        positions = positions.to(inputs_embeds.device)

        hidden_states = inputs_embeds + positions

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        return hidden_states
    
    # Apply monkey patch
    kosmos2_module.Kosmos2TextTransformer.forward_embedding = patched_forward_embedding

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
    ds = ds.class_encode_column("answer").train_test_split(test_size=0.1, stratify_by_column='answer')
    train_ds_1 = RealworldDataset(ds['train'])
    val_ds_1 = RealworldDataset(ds['test'])
    
    # A-OKVQA
    train_ds_2 = load_dataset("HuggingFaceM4/A-OKVQA", split="train")
    val_ds_2 = load_dataset("HuggingFaceM4/A-OKVQA", split="validation")
    val_ds_2 = val_ds_2.shuffle().select(range(100))
    train_ds_2 = AokvqaDataset(train_ds_2)
    val_ds_2 = AokvqaDataset(val_ds_2)
    
    # DACON official
    train_ds_4 = load_dataset("json", data_files="eg/train_aug.jsonl", split='train')
    train_ds_4 = train_ds_4.train_test_split(test_size=0.1)
    val_ds_4 = DACONDataset(train_ds_4['test'])
    train_ds_4 = DACONDataset(train_ds_4['train'])
    
    # VMC Dataset 
    ds = load_from_disk("datasets/VMC/VMC_aug")
    class_labels = ClassLabel(names=["A", "B", "C", "D"])
    ds = ds.cast_column("answer", class_labels)
    ds = ds.train_test_split(test_size=0.1, stratify_by_column='answer')
    train_ds_3, val_ds_3 = ds['train'], ds['test']
    train_ds_3 = VMCDataset(train_ds_3)
    val_ds_3 = VMCDataset(val_ds_3)
    
    # Stanford Dataset
    ds = load_dataset("csv", data_files="datasets/stanford_img_para_caption/stanford_df_rectified.csv", split='train')
    train_ds_5 = ds.filter(lambda x: x['train'])
    val_ds_5 = ds.filter(lambda x: x['test'])
    val_ds_5 = val_ds_5.shuffle().select(range(100))
    train_ds_5 = StanfordDataset(train_ds_5)
    val_ds_5 = StanfordDataset(val_ds_5)
    
    entire_train = [train_ds_1, train_ds_2, train_ds_3, train_ds_4, train_ds_5]
    entire_val = [val_ds_1, val_ds_2, val_ds_3, val_ds_4, val_ds_5]
    
    train_ds = ConcatDataset(entire_train)
    val_ds = ConcatDataset(entire_val)
    print(f"Total number of train samples: {len(train_ds)}")
    print(f"Total number of validation samples: {len(val_ds)}")
    
    train_weights = [[1.0/len(d)] * len(d) for d in entire_train ]
    train_weights = list(chain.from_iterable(train_weights))
    assert len(train_weights) == len(train_ds), "ERROR: weight size and train dataset size mismatch."
    
    val_weights = [[1.0/len(d)] * len(d) for d in entire_val ]
    val_weights = list(chain.from_iterable(val_weights))
    
    # 메모리 정리
    del ds, train_ds_1, train_ds_2, train_ds_3, train_ds_4, train_ds_5, val_ds_1, val_ds_2, val_ds_3, val_ds_4, val_ds_5, entire_train, entire_val
    torch.cuda.empty_cache() # GPU 캐시 비우기
    gc.collect()             # Python 가비지 컬렉터 실행
    
    # ------------------------------ Qunatization Config -----------------------------------
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,                   # QLoRA NF4 적용
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, # 연산 시 bfloat16 사용 (권장)
        )
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
    
    # --------------------- 모델 선언 -----------------------
    model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224", 
                                                            device_map='auto', 
                                                            quantization_config=quantization_config, 
                                                            torch_dtype=torch.bfloat16, 
                                                            attn_implementation="sdpa",)
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224", add_eos_token=True)  
    model = get_peft_model(model, lora_config)
    
    # 모델 정보
    print(f"모델 정보:")
    model.print_trainable_parameters()
    
    # -------------------------------------------- Description 훈련 Phase 1 -------------------------------------------
    data_collator = DataCollator_1(processor, model_dtype=torch.bfloat16)
    
    batch_size = 32

    training_args = SFTConfig(
        report_to='none',
        # run_name = 'run-0729-SOTA' + "-phase_1_QLoRA_NF4",
        output_dir="Model/kosmos2/sota/phase_1",
        overwrite_output_dir=True,
        num_train_epochs=2,     # epochs
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=4,
        eval_do_concat_batches=True,
        eval_accumulation_steps=2,
        gradient_accumulation_steps=128 // batch_size,
        warmup_ratio=0.03,
        learning_rate=2e-4,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        lr_scheduler_kwargs = {"num_cycles": 2.5},
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy="epoch",
        save_total_limit=1,
        metric_for_best_model="meteor",
        greater_is_better=False,
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
        train_weights=train_weights,
        eval_weights=val_weights,
        compute_metrics= lambda eval_pred: compute_meteor(eval_pred, processor.tokenizer),
    )
    
    print("✅ 1st Training start.")
    trainer.train()
    print("✅ 1st Training done.")
    
    # # 모델 수동 저장
    final_save_dir = os.path.join(training_args.output_dir, "best")
    os.makedirs(final_save_dir, exist_ok=True)
    trainer.model.save_pretrained(final_save_dir)     # PEFT 어댑터 (LoRA 가중치)와 PEFT 설정만 저장
    print(f"PEFT adapter saved to {final_save_dir}")
    processor.save_pretrained(final_save_dir)     # processor 저장
    print(f"Processor saved to {final_save_dir}")
    
    # 베스트 모델 로드
    model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224", 
                                                            quantization_config=quantization_config, 
                                                            torch_dtype=torch.bfloat16, 
                                                            device_map="auto" )   # 베이스모델 로드
    model = PeftModel.from_pretrained(model, final_save_dir, torch_dtype=torch.bfloat16, device_map="auto", is_trainable=True )        # LoRA weight 적용된 모델 로드
    processor = AutoProcessor.from_pretrained(final_save_dir)
    
    # 메모리 정리
    torch.cuda.empty_cache() # GPU 캐시 비우기
    gc.collect()             # Python 가비지 컬렉터 실행
    
    # # --------------------------------------------------------- Description 훈련 Phase 2 ---------------------------------------------------------
    # Recap COCO
    ds = load_dataset("UCSC-VLAA/Recap-COCO-30K", split="train")
    ds = ds.train_test_split(test_size=200)
    train_ds = RecapCOCODataset(ds['train'])
    val_ds = RecapCOCODataset(ds['test'])
    
    print(f"Total number of train samples: {len(train_ds)}")
    print(f"Total number of validation samples: {len(val_ds)}")
    
    # 메모리 정리
    del ds, trainer, training_args
    torch.cuda.empty_cache() # GPU 캐시 비우기
    gc.collect()             # Python 가비지 컬렉터 실행
    
    data_collator = DataCollator_1(processor, model_dtype=torch.bfloat16)
    
    batch_size = 28
    training_args = SFTConfig(
        report_to='none',
        # run_name = 'run-0729-SOTA' + "-phase_2_QLoRA_NF4",
        output_dir="Model/kosmos2/sota/phase_2",
        overwrite_output_dir=True,
        num_train_epochs=2,     # epochs
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=4,
        eval_do_concat_batches=True,
        eval_accumulation_steps=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        learning_rate=1e-4,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        lr_scheduler_kwargs = {"num_cycles": 2.5},
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy="epoch",
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
    
    # # 모델 수동 저장
    final_save_dir = os.path.join(training_args.output_dir, "best")
    trainer.model.save_pretrained(final_save_dir)     # PEFT 어댑터 (LoRA 가중치)와 PEFT 설정만 저장
    print(f"PEFT adapter saved to {final_save_dir}")
    processor.save_pretrained(final_save_dir)     # processor 저장
    print(f"Processor saved to {final_save_dir}")

    print('✅ Training All Done.')


if __name__ == "__main__":
    main()