import pandas as pd
import os
from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision.transforms.functional import pil_to_tensor, resize
import torch
import argparse

from utils import read_prompt_file

parser = argparse.ArgumentParser()
parser.add_argument('--images_folder', type=str, default='evaluation_massive/celebA_342_ti/<celebA>/gen_prompt_actions')
parser.add_argument('--prompt_file', type=str, default='prompts/gen_prompt_actions.csv')
parser.add_argument('--anchor_image_path', type=str, default='celebA/342/416.jpg')
parser.add_argument('--num_images', type=int, default=100)
parser.add_argument('--info', type=str, default='416')
parser.add_argument('--output_dir', type=str, default='semantic_drift')
parser.add_argument('--sub_folder', type=str, default='None')

args = parser.parse_args()

metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")
metric.to('cuda')

os.makedirs(args.output_dir, exist_ok=True)

images_folder = args.images_folder
prompt_file = args.prompt_file
anchor_image_path = args.anchor_image_path
num_images = args.num_images
info = args.info

prompt_name = prompt_file.split('/')[-1].split('.')[0]
exp_name = images_folder.split('/')[1] # celebA_342_ti
output_file = f'{args.output_dir}/clip_alignment_with_anchor_image_{exp_name}_{prompt_name}_{info}.csv'

def process_anchor_image(anchor_image_path):
    anchor_image = Image.open(anchor_image_path)

    # Convert RGBA to RGB 
    if anchor_image.mode == 'RGBA':
        anchor_image = anchor_image.convert('RGB')

    anchor_image = pil_to_tensor(anchor_image)
    anchor_image = resize(anchor_image, 224, antialias=True)
    anchor_image = anchor_image.unsqueeze(0)
    anchor_image = anchor_image.to('cuda')
    assert anchor_image.shape[1] == 3, f'Expected 3 channels, but got {anchor_image.shape[1]} channels'
    assert anchor_image.shape == (1, 3, 224, 224), f'anchor_image.shape is not (1, 3, 224, 224), but {anchor_image.shape}'
    return anchor_image

anchor_image = process_anchor_image(anchor_image_path)
print('anchor_image.shape', anchor_image.shape)

def calculate_clip_image_prompt_score(images_folder, prompt_file, num_images):

    prompts = read_prompt_file(prompt_file)

    for prompt in prompts:
        print(prompt)

    # sanity check
    # count the number of images in the images_folder
    print(f'Number of images: {len([f for f in os.listdir(images_folder) if f.endswith(".png")])}')

    if len([f for f in os.listdir(images_folder) if f.endswith(".png")]) != num_images * len(prompts):
        print(f'Number of images in the images_folder is not equal to the number of images x number of prompts')
        print(f'Number of images: {len([f for f in os.listdir(images_folder) if f.endswith(".png")])}')
        print(f'Number of prompts: {len(prompts)}')
        print(f'Number of images x number of prompts: {num_images * len(prompts)}')
        print(f'Images folder: {images_folder}')
        # return None


    new_prompts = prompts       

    # calculate the clip alignment score between the prompt and the corresponding image in the images_folder
    # naming rule of the images `{prompt_index}_{image_index}.png`
    # write the results to a csv file
    results = []
    for prompt_index, prompt in enumerate(new_prompts):
        for image_index in range(num_images):
            image_path = os.path.join(images_folder, f'{prompt_index}_{image_index}.png')
            if not os.path.exists(image_path):
                print(f'Image {image_path} does not exist')
                continue
            image = Image.open(image_path)
            image = pil_to_tensor(image)

            image = resize(image, 224, antialias=True)
            image = image.unsqueeze(0)
            image = image.to('cuda')

            with torch.no_grad():
                score = metric(image, anchor_image).item()

            print(f'{prompt_index} {image_index} {prompt} {score}')
            results.append({'prompt_index': prompt_index, 'prompt': prompt, 'image_index': image_index, 'score': score})

    return results

if args.sub_folder == 'None':
    results = calculate_clip_image_prompt_score(images_folder, prompt_file, num_images)
    if results is not None:
        results = pd.DataFrame(results)
    results.to_csv(output_file, index=False)

    ## Calculate the same for sub-folders in the images_folder
    for sub_folder in os.listdir(images_folder):
        if os.path.isdir(os.path.join(images_folder, sub_folder)):
            print('--------------------------------')
            print(f'Calculating for {sub_folder}')
            results = calculate_clip_image_prompt_score(os.path.join(images_folder, sub_folder), prompt_file, num_images)
            if results is not None:
                results = pd.DataFrame(results)
                results.to_csv(f'{args.output_dir}/clip_alignment_with_anchor_image_{exp_name}_{prompt_name}_{sub_folder}_{info}.csv', index=False)
            else:
                print(f'No results for {sub_folder}')

else:
    assert args.sub_folder in os.listdir(images_folder), f'{args.sub_folder} is not a sub-folder in {images_folder}'
    assert os.path.isdir(os.path.join(images_folder, args.sub_folder)), f'{args.sub_folder} is not a directory'
    results = calculate_clip_image_prompt_score(os.path.join(images_folder, args.sub_folder), prompt_file, num_images)
    if results is not None:
        results = pd.DataFrame(results)
        results.to_csv(f'{args.output_dir}/clip_alignment_with_anchor_image_{exp_name}_{prompt_name}_{args.sub_folder}_{info}.csv', index=False)
    else:
        print(f'No results for {args.sub_folder}')