import pandas as pd
import os
from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision.transforms.functional import pil_to_tensor, resize
import torch
import argparse
from myutils import str2bool

parser = argparse.ArgumentParser()
parser.add_argument('--images_folder', type=str, default='evaluation_massive/celebA_342_ti/<celebA>/gen_prompt_actions')
parser.add_argument('--prompt_file', type=str, default='prompts/gen_prompt_actions.csv')
parser.add_argument('--num_images', type=int, default=100)
parser.add_argument('--use_custom_prompt', type=str, default='objectA')
parser.add_argument('--output_dir', type=str, default='semantic_drift')
parser.add_argument('--info', type=str, default='use_custom_prompt_remove_subject_and_glasses')
parser.add_argument('--sub_folder', type=str, default='None')

args = parser.parse_args()

metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")
metric.to('cuda')

os.makedirs(args.output_dir, exist_ok=True)

images_folder = args.images_folder
prompt_file = args.prompt_file
num_images = args.num_images
use_custom_prompt = args.use_custom_prompt
info = args.info

prompt_name = prompt_file.split('/')[-1].split('.')[0]
exp_name = images_folder.split('/')[1] # celebA_342_ti
output_file = f'{args.output_dir}/clip_alignment_scores_{exp_name}_{prompt_name}_{info}.csv'

def calculate_clip_image_prompt_score(images_folder, prompt_file, prompt_name, num_images, use_custom_prompt=False):

    assert os.path.isfile(prompt_file), f'Prompt file {prompt_file} does not exist'
    assert prompt_file.endswith('.csv'), f'Prompt file {prompt_file} is not a csv file'

    # read from the csv file
    df = pd.read_csv(prompt_file)
    
    # Print column names for debugging
    print(f"Available columns in CSV: {df.columns.tolist()}")
    
    # Get prompts from the first column
    prompts = df.iloc[:, 0].tolist()
    
    # Get custom prompts safely, handling different possible column names
    if 'custom_prompt_A' in df.columns:
        custom_prompts_A = df['custom_prompt_A'].tolist()
    else:
        custom_prompts_A = df.iloc[:, 1].tolist()  # Use second column
        
    if 'custom_prompt_B' in df.columns:
        custom_prompts_B = df['custom_prompt_B'].tolist()
    else:
        custom_prompts_B = df.iloc[:, 2].tolist()  # Use third column
        
    if 'custom_prompt_C' in df.columns:
        custom_prompts_C = df['custom_prompt_C'].tolist()
    else:
        custom_prompts_C = ["NA"] * len(prompts)  # Use fourth column if exists

    if 'full_prompt' in df.columns:
        full_prompts = df['full_prompt'].tolist()
    else:
        full_prompts = prompts
    
    for prompt, cA, cB, cC, full_prompt in zip(prompts, custom_prompts_A, custom_prompts_B, custom_prompts_C, full_prompts):
        print(prompt, cA, cB, cC, full_prompt)

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

    new_prompts = []
    if use_custom_prompt == 'objectA':
        assert len(custom_prompts_A) == len(prompts), f'Number of custom prompts is not equal to the number of prompts'
        for prompt, cA in zip(prompts, custom_prompts_A):
            new_prompt = cA
            new_prompts.append(new_prompt)
    elif use_custom_prompt == 'objectB':
        assert len(custom_prompts_B) == len(prompts), f'Number of custom prompts is not equal to the number of prompts'
        for prompt, cB in zip(prompts, custom_prompts_B):
            new_prompt = cB
            new_prompts.append(new_prompt)
    elif use_custom_prompt == 'objectC':
        assert len(custom_prompts_C) == len(prompts), f'Number of custom prompts is not equal to the number of prompts'
        for prompt, cC in zip(prompts, custom_prompts_C):
            new_prompt = cC
            new_prompts.append(new_prompt)
    elif use_custom_prompt == 'full_prompt':
        new_prompts = full_prompts      
    else:
        raise ValueError(f'use_custom_prompt must be either "objectA" or "objectB" or "objectC" or "objectD" or "full_prompt"')

    # calculate the clip alignment score between the prompt and the corresponding image in the images_folder
    # naming rule of the images `{prompt_index}_{image_index}.png`
    # write the results to a csv file
    results = []
    for prompt_index, prompt in enumerate(new_prompts):
        for image_index in range(num_images):
            image_path = os.path.join(images_folder, f'{prompt_index}_{image_index}.png')
            image = Image.open(image_path)
            image = pil_to_tensor(image)

            image = resize(image, 224, antialias=True)
            image = image.unsqueeze(0)
            image = image.to('cuda')

            with torch.no_grad():
                score = metric(image, prompt).item()

            print(f'{prompt_index} {image_index} {prompt} {score}')
            results.append({'prompt_index': prompt_index, 'prompt': prompt, 'image_index': image_index, 'score': score})

    return results

if args.sub_folder == 'None':
    results = calculate_clip_image_prompt_score(images_folder, prompt_file, prompt_name, num_images, use_custom_prompt)
    if results is not None:
        results = pd.DataFrame(results)
        results.to_csv(output_file, index=False)

    ## Calculate the same for sub-folders in the images_folder
    for sub_folder in os.listdir(images_folder):
        if os.path.isdir(os.path.join(images_folder, sub_folder)):
            print('--------------------------------')
            print(f'Calculating for {sub_folder}')
            results = calculate_clip_image_prompt_score(os.path.join(images_folder, sub_folder), prompt_file, prompt_name, num_images, use_custom_prompt)
            if results is not None:
                results = pd.DataFrame(results)
                results.to_csv(f'{args.output_dir}/clip_alignment_scores_{exp_name}_{prompt_name}_{sub_folder}_{info}.csv', index=False)
            else:
                print(f'No results for {sub_folder}')

else:
    assert args.sub_folder in os.listdir(images_folder), f'{args.sub_folder} is not a sub-folder in {images_folder}'
    assert os.path.isdir(os.path.join(images_folder, args.sub_folder)), f'{args.sub_folder} is not a directory'
    results = calculate_clip_image_prompt_score(os.path.join(images_folder, args.sub_folder), prompt_file, prompt_name, num_images, use_custom_prompt)
    if results is not None:
        results = pd.DataFrame(results)
        results.to_csv(f'{args.output_dir}/clip_alignment_scores_{exp_name}_{prompt_name}_{args.sub_folder}_{info}.csv', index=False)
    else:
        print(f'No results for {args.sub_folder}')
