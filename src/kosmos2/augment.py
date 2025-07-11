import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import os
import json
from tqdm import tqdm


IMAGENET_MEAN = (0.5, 0.5, 0.5)
IMAGENET_STD = (0.5, 0.5, 0.5)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=10, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_data, input_size=384, max_num=10):
    image = Image.open(image_data).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values



def main(mode):
    model_path = 'LiAutoAD/Ristretto-3B'
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    generation_config = dict(min_new_tokens=20, max_new_tokens=128, repetition_penalty=1.2, length_penalty = 1.1, do_sample=False, num_beams=3)
    
    
    # visual 7w의 경우
    if mode == 'visual7w':
        ds = load_dataset("json", data_files="datasets/visual7w/dataset_v7w_telling.json", split='train[17457:]')
        ds = ds['images']
        output_jsonl_path = os.path.join("datasets/visual7w","visual7w_augmented_with_descriptions.jsonl")

        for idx, sample in tqdm(enumerate(ds)):
            image_path = os.path.join('datasets/visual7w/images', sample['filename'])
            pixel_values = load_image(image_path).to(torch.bfloat16).cuda()

            question = '<image>         Describe the image in detail.'
            response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
            # Image._show(Image.open(image_path))
            print(response)
            sample['description'] = response

            with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # VMC bench의 경우
    elif mode == 'VMC':
        ds = load_dataset("suyc21/VMCBench", split='dev')
        ds.remove_columns(["index", "category"])
        output_path = os.path.join("datasets/VMC/VMC_aug")

        def add_description_to_sample(sample):
            image = sample['image']
            transform = build_transform(input_size=384)
            images = dynamic_preprocess(image, image_size=384, use_thumbnail=True, max_num=10)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()

            question = '<image>         Describe the image in detail.'
            response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
            print(response)
            sample['description'] = response
            return sample
        
        ds = ds.map(add_description_to_sample, batched=False, desc="Adding descriptions")
        ds.save_to_disk(output_path)
                
    # Realworld QA의 경우            
    elif mode == 'realworld':
        ds = load_dataset("xai-org/RealworldQA", split='test')
        output_path = os.path.join("datasets/Realworld/Realworld_aug")

        def add_description_to_sample(sample):
            image = sample['image']
            transform = build_transform(input_size=384)
            images = dynamic_preprocess(image, image_size=384, use_thumbnail=True, max_num=10)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()

            question = '<image>         Describe the image in detail.'
            response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
            print(response)
            sample['description'] = response
            return sample
        
        ds = ds.map(add_description_to_sample, batched=False, desc="Adding descriptions")
        ds.save_to_disk(output_path)



if __name__ == '__main__':
    main('realworld')     # VMC, realworld, visual7w