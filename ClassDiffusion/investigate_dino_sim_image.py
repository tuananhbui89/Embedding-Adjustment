import pandas as pd
import os
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from torchvision.transforms.functional import pil_to_tensor, resize
import torch
import argparse

from myutils import read_prompt_file

parser = argparse.ArgumentParser()
parser.add_argument('--images_folder', type=str, default='evaluation_massive/celebA_342_ti/<celebA>/gen_prompt_actions')
parser.add_argument('--prompt_file', type=str, default='prompts/gen_prompt_actions.csv')
parser.add_argument('--anchor_image_path', type=str, default='celebA/342/416.jpg')
parser.add_argument('--num_images', type=int, default=100)
parser.add_argument('--info', type=str, default='416')
parser.add_argument('--output_dir', type=str, default='semantic_drift')
parser.add_argument('--sub_folder', type=str, default='None')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

os.makedirs(args.output_dir, exist_ok=True)

images_folder = args.images_folder
prompt_file = args.prompt_file
anchor_image_path = args.anchor_image_path
num_images = args.num_images
info = args.info

prompt_name = prompt_file.split('/')[-1].split('.')[0]
exp_name = images_folder.split('/')[1] # celebA_342_ti
output_file = f'{args.output_dir}/dino_alignment_with_anchor_image_{exp_name}_{prompt_name}_{info}.csv'

def process_image(image_path):
    image = Image.open(image_path)

    # Convert RGBA to RGB 
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # image = pil_to_tensor(image)
    # image = resize(image, 224, antialias=True)
    # image = image.unsqueeze(0)
    # image = image.to('cuda')
    # assert image.shape[1] == 3, f'Expected 3 channels, but got {image.shape[1]} channels'
    # assert image.shape == (1, 3, 224, 224), f'image.shape is not (1, 3, 224, 224), but {image.shape}'
    return image


def dino_embedding(image):
    input = processor(images=image, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**input)
    feature = outputs.last_hidden_state
    return feature.mean(dim=1)

def dino_similarity(image, anchor_image_embedding):
    image_embedding = dino_embedding(image)
    assert image_embedding.shape == anchor_image_embedding.shape, f'image_embedding.shape is not equal to anchor_image_embedding.shape, but {image_embedding.shape} and {anchor_image_embedding.shape}'
    return torch.nn.functional.cosine_similarity(image_embedding, anchor_image_embedding, dim=1)

anchor_image = process_image(anchor_image_path)
anchor_image_embedding = dino_embedding(anchor_image)
print('anchor_image.size', anchor_image.size)

def calculate_dino_image_image_score(images_folder, prompt_file, num_images):

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
            image = process_image(image_path)

            with torch.no_grad():
                score = dino_similarity(image, anchor_image_embedding).item()

            print(f'{prompt_index} {image_index} {prompt} {score}')
            results.append({'prompt_index': prompt_index, 'prompt': prompt, 'image_index': image_index, 'score': score})

    return results

if args.sub_folder == 'None':
    results = calculate_dino_image_image_score(images_folder, prompt_file, num_images)
    if results is not None:
        results = pd.DataFrame(results)
    results.to_csv(output_file, index=False)

    ## Calculate the same for sub-folders in the images_folder
    for sub_folder in os.listdir(images_folder):
        if os.path.isdir(os.path.join(images_folder, sub_folder)):
            print('--------------------------------')
            print(f'Calculating for {sub_folder}')
            results = calculate_dino_image_image_score(os.path.join(images_folder, sub_folder), prompt_file, num_images)
            if results is not None:
                results = pd.DataFrame(results)
                results.to_csv(f'{args.output_dir}/dino_alignment_with_anchor_image_{exp_name}_{prompt_name}_{sub_folder}_{info}.csv', index=False)
            else:
                print(f'No results for {sub_folder}')

else:
    assert args.sub_folder in os.listdir(images_folder), f'{args.sub_folder} is not a sub-folder in {images_folder}'
    assert os.path.isdir(os.path.join(images_folder, args.sub_folder)), f'{args.sub_folder} is not a directory'
    results = calculate_dino_image_image_score(os.path.join(images_folder, args.sub_folder), prompt_file, num_images)
    if results is not None:
        results = pd.DataFrame(results)
        results.to_csv(f'{args.output_dir}/dino_alignment_with_anchor_image_{exp_name}_{prompt_name}_{args.sub_folder}_{info}.csv', index=False)
    else:
        print(f'No results for {args.sub_folder}')