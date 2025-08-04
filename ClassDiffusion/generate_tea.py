import torch
import argparse
import os
import numpy as np
import random

from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import DiffusionPipeline
from diffusers import UNet2DConditionModel

from transformers import CLIPTextModel, CLIPTokenizer

from myutils import set_seed, read_prompt_file

from my_pipeline import assign_func_call
import my_pipeline as my_pipelines

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--concept_name", type=str, default="<cathy>")
    parser.add_argument("--output_dir", type=str, default="evaluation_folder")
    parser.add_argument("--model_path", type=str, default="outputs/cathy_4")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--prompt_file", type=str, default="None")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_images", type=int, default=1)

    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--target_name", type=str, default="man")

    args = parser.parse_args()
    set_seed(args.seed)
    return args


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

def main(args):

    pipeline = load_my_pipeline("runwayml/stable-diffusion-v1-5")

    # Disable NSFW detector
    pipeline.safety_checker = None

    pipeline = pipeline.to("cuda")

    # Load the trained custom diffusion weights
    assert os.path.exists(args.model_path + "/pytorch_custom_diffusion_weights.bin")
    pipeline.unet.load_attn_procs(
        args.model_path, 
        weight_name="pytorch_custom_diffusion_weights.bin"  # or .safetensors
    )

    # Load the trained textual inversion embeddings
    assert os.path.exists(args.model_path + "/" + args.concept_name + ".bin")
    pipeline.load_textual_inversion(
        args.model_path, 
        weight_name=args.concept_name + ".bin"  # Replace <new1> with your modifier token
    )
    
    prompts = read_prompt_file(args.prompt_file)
    generator = torch.Generator("cuda").manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    for i, prompt in enumerate(prompts):
        for j in range(args.num_images):
            print(i, j, "prompt: ", prompt.format(args.concept_name), "target: ", prompt.format(args.target_name))
            image = pipeline(prompt.format(args.target_name), args.rho, args.alpha, prompt.format(args.concept_name), num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, generator=generator).images[0]
            image.save(f"{args.output_dir}/{i}_{j}.png")
  

if __name__ == "__main__":
    args = parse_args()
    main(args)
