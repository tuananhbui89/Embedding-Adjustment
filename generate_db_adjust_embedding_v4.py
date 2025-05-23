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

from utils import set_seed, read_prompt_file

from my_pipelines_adjust_embedding import assign_func_call
import my_pipelines_adjust_embedding as my_pipelines

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--concept_name", type=str, default="<cathy>")
    parser.add_argument("--output_dir", type=str, default="evaluation_folder")
    parser.add_argument("--model_path", type=str, default="outputs/cathy_4")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--lora_path", type=str, default="outputs/cathy_4/lora_weights.safetensors")
    parser.add_argument("--max_train_steps", type=int, default=3000)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--prompt_file", type=str, default="None")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--target_name", type=str, default="man")

    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )

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

def main():
    args = parse_args()
    print(args)

    generator = torch.Generator("cuda").manual_seed(args.seed)

    # create output directory
    prompt_name = args.prompt_file.split("/")[-1].split(".")[0]
    output_dir = f"{args.output_dir}/{args.concept_name}/{prompt_name}"
    os.makedirs(output_dir, exist_ok=True)

    assert os.path.exists(args.prompt_file), f"Prompt file {args.prompt_file} does not exist"
    prompts = read_prompt_file(args.prompt_file)

    # unet = UNet2DConditionModel.from_pretrained(
    #     args.model_path, subfolder="unet"
    # )
    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.model_path, subfolder="text_encoder"
    # )

    # pipeline = DiffusionPipeline.from_pretrained(args.model_path, unet=unet, text_encoder=text_encoder)
    # pipeline.to("cuda")

    pipeline = load_my_pipeline(args.model_path)

    # Disable NSFW detector
    pipeline.safety_checker = None
    
    # pipeline.load_lora_weights(args.lora_path, weight_name="pytorch_lora_weights.safetensors")

    if args.train_text_encoder:
        pipeline.load_lora_weights(args.lora_path, weight_name="pytorch_lora_weights.safetensors")
    else:
        # load lora weights but specific to the unet - that works
        pipeline.unet.load_lora_adapter(args.lora_path, weight_name="pytorch_lora_weights.safetensors", prefix="unet")


    for i, prompt in enumerate(prompts):
        for j in range(args.num_images):
            print(i, j, "prompt: ", prompt.format(args.concept_name))
            if args.guidance_scale is None:
                image = pipeline(prompt.format(args.target_name), args.rho, args.alpha, prompt.format(args.concept_name), num_inference_steps=args.num_inference_steps, generator=generator).images[0]
            else:
                image = pipeline(prompt.format(args.target_name), args.rho, args.alpha, prompt.format(args.concept_name), num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, generator=generator).images[0]
            image.save(f"{output_dir}/{i}_{j}.png")

if __name__ == "__main__":
    main()

