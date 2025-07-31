import os
import argparse

import torch
from PIL import Image

from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from templates.templates import inference_templates
from tea import adjust_norm_and_slerp
from my_pipeline import assign_func_call
import my_pipeline as my_pipelines

import math
from utils import read_prompt_file

"""
Inference script for generating batch results
"""


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--prompt",
        type=str,
        help="input a single text prompt for generation",
    )
    parser.add_argument(
        "--template_name",
        type=str,
        help="select a batch of text prompts from templates.py for generation",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="absolute path to the folder that contains the trained results",
    )
    parser.add_argument(
        "--placeholder_string",
        type=str,
        default="<R>",
        help="place holder string of the relation prompt",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="number of samples to generate for each prompt",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="scale for classifier-free guidance",
    )
    parser.add_argument(
        "--only_load_embeds",
        action="store_true",
        default=False,
        help="If specified, the experiment folder only contains the relation prompt, but does not contain the entire folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference",
        help="output directory",
    )

    # New arguments for TEA 
    parser.add_argument(
        "--rho",
        type=float,
        default=0.2,
        help="rho for TEA",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="alpha for TEA",
    )
    parser.add_argument(
        "--target_word",
        type=str,
        default="handshake",
        help="target word for TEA",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompts_carved_by.csv",
        help="prompt file",
    )
    args = parser.parse_args()
    return args


def make_image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def load_my_pipeline(model_path):
    assign_func_call(StableDiffusionPipeline)
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")

    pipeline = my_pipelines.StableDiffusionPipeline.from_pretrained(
        model_path,
        unet=unet,
        text_encoder=text_encoder,
        safety_checker=None,
        weight_dtype=torch.float32,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to("cuda")
    return pipeline

def main():
    args = parse_args()

    # create inference pipeline
    assert args.only_load_embeds

    embed_path = os.path.join(args.model_id, 'learned_embeds.bin')
    learned_embeds = torch.load(embed_path)
    
    # pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False).to("cuda")
    pipe = load_my_pipeline("runwayml/stable-diffusion-v1-5")

    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    # keep original embeddings as reference
    orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()

    # Add the placeholder token in tokenizer
    tokenizer.add_tokens(args.placeholder_string)
    text_encoder.get_input_embeddings().weight.data = torch.cat((orig_embeds_params, orig_embeds_params[0:1]))
    text_encoder.resize_token_embeddings(len(tokenizer)) 

    # Let's make sure we don't update any embedding weights besides the newly added token
    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_string)
    index_no_updates = torch.arange(len(tokenizer)) != placeholder_token_id
    text_encoder.get_input_embeddings().weight.data[index_no_updates] = orig_embeds_params
    text_encoder.get_input_embeddings().weight.data[placeholder_token_id] = learned_embeds[args.placeholder_string]

    # make directory to save images
    # image_root_folder = os.path.join(args.model_id, 'inference')
    image_root_folder = args.output_dir
    os.makedirs(image_root_folder, exist_ok = True)

    if args.prompt is None and args.template_name is None:
        raise ValueError("please input a single prompt through'--prompt' or select a batch of prompts using '--template_name'.")

    # single text prompt
    if args.prompt is not None:
        prompt_list = [args.prompt]
    else:
        prompt_list = []

    if args.template_name is not None:
        # read the selected text prompts for generation
        prompt_list.extend(inference_templates[args.template_name])

    prompt_list = read_prompt_file(args.prompt_file)

    for prompt in prompt_list:
        # insert relation prompt <R>
        org_prompt = prompt.lower().replace("<r>", "<R>").format(args.placeholder_string)
        target_prompt = prompt.lower().replace("<r>", "<R>").format(args.target_word)
        print(f"prompt: {org_prompt}")
        print(f"target_prompt: {target_prompt}")

        # make sub-folder
        image_folder = os.path.join(image_root_folder, org_prompt, 'samples')
        os.makedirs(image_folder, exist_ok = True)

        # batch generation
        all_images = []
        for idx in range(args.num_samples):
            images = pipe(target_prompt, args.rho, args.alpha, org_prompt, num_inference_steps=50, guidance_scale=args.guidance_scale, num_images_per_prompt=1).images
            images[0].save(os.path.join(image_folder, f"{str(idx).zfill(4)}.png"))
            all_images.append(images[0])

        

        # save a grid of images
        image_grid = make_image_grid(all_images, rows=5, cols=math.ceil(args.num_samples/5))
        image_grid_path = os.path.join(image_root_folder, org_prompt, f'{org_prompt}.png')
        image_grid.save(image_grid_path)
        print(f'saved to {image_grid_path}')

        # clear the cache
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
