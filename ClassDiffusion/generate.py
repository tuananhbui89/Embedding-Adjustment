import torch
import argparse
import os
import numpy as np
import random

from diffusers import DiffusionPipeline
from myutils import set_seed, read_prompt_file

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

    args = parser.parse_args()
    set_seed(args.seed)
    return args

def main(args):

    # Load the base pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16,
    ).to("cuda")

    # Load the trained custom diffusion weights
    pipeline.unet.load_attn_procs(
        args.model_path, 
        weight_name="pytorch_custom_diffusion_weights.bin"  # or .safetensors
    )

    # Load the trained textual inversion embeddings
    pipeline.load_textual_inversion(
        args.model_path, 
        weight_name=args.concept_name + ".bin"  # Replace <new1> with your modifier token
    )
    
    prompts = read_prompt_file(args.prompt_file)
    generator = torch.Generator("cuda").manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    for i, prompt in enumerate(prompts):
        for j in range(args.num_images):
            print(i, j, "prompt: ", prompt.format(args.concept_name))
            image = pipeline(prompt.format(args.concept_name), num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, generator=generator).images[0]
            image.save(f"{args.output_dir}/{i}_{j}.png")
  

if __name__ == "__main__":
    args = parse_args()
    main(args)
