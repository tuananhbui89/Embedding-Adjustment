import torch
import argparse
import os
import numpy as np
import random

from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import DiffusionPipeline
from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel

from transformers import CLIPTextModel, CLIPTokenizer

import my_pipelines

from my_pipelines import SimpleProjector, SimpleProjectorOneLayer
from my_pipelines import assign_func_call
# from my_pipelines import StableDiffusionPipeline
from utils import str2bool
from utils import set_seed, read_prompt_file

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--concept_name", type=str, default="<cathy>")
    parser.add_argument("--output_dir", type=str, default="evaluation_folder")
    parser.add_argument("--model_path", type=str, default="outputs/cathy_4")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--projector_path", type=str, default="outputs/cathy_4/projector.pth")
    parser.add_argument("--method", type=str, default="gti")
    parser.add_argument("--projector_type", type=str, default="simple_projector")
    parser.add_argument("--embedding_path", type=str, default="outputs/cathy_4/learned_embeds.bin")
    parser.add_argument("--embedding_path_2", type=str, default="None")
    parser.add_argument("--max_train_steps", type=int, default=3000)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--prompt_file", type=str, default="None")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--start_steps", type=int, default=0)
    parser.add_argument("--unet_path", type=str, default="None")
    parser.add_argument("--start_prompt_index", type=int, default=0)
    
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )

    args = parser.parse_args()
    set_seed(args.seed)
    return args

def main_gen_from_prompt(args):
    generator = torch.Generator("cuda").manual_seed(args.seed)

    pipeline = load_standard_pipeline(args.model_path)

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    assert type(args.prompt_file) == str, f"Prompt file {args.prompt_file} is not a string"
    assert not args.prompt_file.endswith(".csv"), f"Prompt file {args.prompt_file} is a csv file"

    prompts = [args.prompt_file]

    for i, prompt in enumerate(prompts):
        for j in range(args.num_images):
            print(i, j, "prompt: ", prompt.format(args.concept_name))
            image = pipeline(prompt.format(args.concept_name), num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, generator=generator).images[0]
            image.save(f"{args.output_dir}/{i}_{j}.png")



def main_standard(args):
    generator = torch.Generator("cuda").manual_seed(args.seed)

    pipeline = load_standard_pipeline(args.model_path)
    # pipeline.load_textual_inversion(f"sd-concepts-library/{args.concept_name}")

    # create output directory
    prompt_name = args.prompt_file.split("/")[-1].split(".")[0]
    output_dir = f"{args.output_dir}/{args.concept_name}/{prompt_name}"
    os.makedirs(output_dir, exist_ok=True)

    assert os.path.exists(args.prompt_file), f"Prompt file {args.prompt_file} does not exist"

    prompts = read_prompt_file(args.prompt_file)

    for i, prompt in enumerate(prompts):
        for j in range(args.num_images):
            print(i, j, "prompt: ", prompt.format(args.concept_name))
            image = pipeline(prompt.format(args.concept_name), num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, generator=generator).images[0]
            image.save(f"{output_dir}/{i}_{j}.png")


def main_ti(args):
    generator = torch.Generator("cuda").manual_seed(args.seed)

    pipeline = load_pipeline(args.model_path, args.embedding_path, args.concept_name)
    # pipeline.load_textual_inversion(f"sd-concepts-library/{args.concept_name}")

    # create output directory
    prompt_name = args.prompt_file.split("/")[-1].split(".")[0]
    output_dir = f"{args.output_dir}/{args.concept_name}/{prompt_name}"
    os.makedirs(output_dir, exist_ok=True)

    assert os.path.exists(args.prompt_file), f"Prompt file {args.prompt_file} does not exist"
    prompts = read_prompt_file(args.prompt_file)

    if args.start_steps == 0:
        for i, prompt in enumerate(prompts):
            if i < args.start_prompt_index:
                continue
            for j in range(args.num_images):
                assert "{}" in prompt, f"Prompt {prompt} does not contain {{}}"
                print(i, j, "prompt: ", prompt.format(args.concept_name))
                image = pipeline(prompt.format(args.concept_name), num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, generator=generator).images[0]
                image.save(f"{output_dir}/{i}_{j}.png")

    for steps in range(args.start_steps, args.max_train_steps):
        # "outputs/cat_toy_ati_v4_0.1/projector-steps-1000.pth"
        if args.embedding_path.endswith(".safetensors"):
            assert "learned_embeds.bin" in args.embedding_path, f"Embedding path {args.embedding_path} does not end with learned_embeds.bin"
            embedding_path = args.embedding_path.replace("learned_embeds.safetensors", f"learned_embeds-steps-{steps}.safetensors")
        else:
            assert "learned_embeds.bin" in args.embedding_path, f"Embedding path {args.embedding_path} does not end with learned_embeds.bin"
            embedding_path = args.embedding_path.replace("learned_embeds.bin", f"learned_embeds-steps-{steps}.bin")

        if not os.path.exists(embedding_path):
            # print(f"Embedding path {embedding_path} does not exist")
            continue

        pipeline = load_pipeline(args.model_path, embedding_path, args.concept_name)
        os.makedirs(f"{output_dir}/{steps}", exist_ok=True)
        
        for i, prompt in enumerate(prompts):
            if i < args.start_prompt_index:
                continue
            for j in range(args.num_images):
                print(i, j, "prompt: ", prompt.format(args.concept_name))
                image = pipeline(prompt=prompt.format(args.concept_name), num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, generator=generator).images[0]
                image.save(f"{output_dir}/{steps}/{i}_{j}.png")



def main_custom_diffusion(args):
    assert "{}" in args.unet_path, f"Unet path {args.unet_path} does not contain {{}}"

    generator = torch.Generator("cuda").manual_seed(args.seed)

    # create output directory
    prompt_name = args.prompt_file.split("/")[-1].split(".")[0]
    output_dir = f"{args.output_dir}/{args.concept_name}/{prompt_name}"
    os.makedirs(output_dir, exist_ok=True)

    assert os.path.exists(args.prompt_file), f"Prompt file {args.prompt_file} does not exist"

    prompts = read_prompt_file(args.prompt_file)

    for steps in range(args.start_steps, args.max_train_steps):

        unet_path = args.unet_path.format(steps)

        if not os.path.exists(unet_path):
            # print(f"Unet path {unet_path} does not exist")
            continue

        pipeline = load_custom_diffusion_pipeline(args.model_path, unet_path, args)
        os.makedirs(f"{output_dir}/{steps}", exist_ok=True)
        
        for i, prompt in enumerate(prompts):
            for j in range(args.num_images):
                print(i, j, "prompt: ", prompt.format(args.concept_name))
                image = pipeline(prompt=prompt.format(args.concept_name), num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, generator=generator).images[0]
                image.save(f"{output_dir}/{steps}/{i}_{j}.png")
        
        del pipeline
        torch.cuda.empty_cache()



def load_standard_pipeline(model_path):
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")

    if "sdxl" in model_path:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            safety_checker=None,
            weight_dtype=torch.float32,
        )
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            safety_checker=None,
            weight_dtype=torch.float32,
        )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to("cuda")
    return pipeline

# def load_pipeline(model_path, embedding_path, concept_name):
#     tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
#     text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")

#     placeholder_token = [f"{concept_name}"]
#     num_added_tokens = tokenizer.add_tokens(placeholder_token)
#     if num_added_tokens == 0:
#         raise ValueError("The tokens are already in the tokenizer")
#     placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
#     text_encoder.resize_token_embeddings(len(tokenizer))

#     learned_embeds = torch.load(embedding_path)
#     token_embeds = text_encoder.get_input_embeddings().weight.data
#     for token, token_id in zip(placeholder_token, placeholder_token_id):
#         token_embeds[token_id] = learned_embeds[token]

#     if "sdxl" in model_path:
#         pipeline = StableDiffusionXLPipeline.from_pretrained(
#             model_path,
#             text_encoder=text_encoder,
#             tokenizer=tokenizer,
#             safety_checker=None,
#             weight_dtype=torch.float32,
#         )
#     else:
#         pipeline = StableDiffusionPipeline.from_pretrained(
#             model_path,
#             text_encoder=text_encoder,
#             tokenizer=tokenizer,
#             safety_checker=None,
#             weight_dtype=torch.float32,
#         )
#     pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
#     pipeline = pipeline.to("cuda")
#     return pipeline


def load_pipeline(model_path, embedding_path, concept_name):
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")

    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")


    if ' ' in concept_name:
        print(f"Concept name {concept_name} contains a space, splitting it into two tokens")
        placeholder_token = concept_name.split(' ')
    else:
        placeholder_token = [f"{concept_name}"]

    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    assert num_added_tokens == len(placeholder_token), f"Number of added tokens {num_added_tokens} does not match the number of placeholder tokens {len(placeholder_token)}"

    if num_added_tokens == 0:
        raise ValueError("The tokens are already in the tokenizer")
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    assert len(placeholder_token_id) == len(placeholder_token), f"Number of placeholder token ids {len(placeholder_token_id)} does not match the number of placeholder tokens {len(placeholder_token)}"
    text_encoder.resize_token_embeddings(len(tokenizer))

    learned_embeds = torch.load(embedding_path)
    token_embeds = text_encoder.get_input_embeddings().weight.data

    print(f"learned_embeds len: {len(learned_embeds)}")
    print(f"token_embeds len: {len(token_embeds)}")

    for i in range(len(placeholder_token_id)):
        # NOTE HERE: We save the embeddings for the two placeholder tokens, but using args.placeholder_token as the key
        token_embeds[placeholder_token_id[i]] = learned_embeds[placeholder_token[0]][i]

    if "sdxl" in model_path:
        raise ValueError("SDXL is not supported for this method")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            safety_checker=None,
            weight_dtype=torch.float32,
        )
    else:
        pipeline = DiffusionPipeline.from_pretrained(
            model_path,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            vae=vae,
            safety_checker=None,
            weight_dtype=torch.float32,
        )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to("cuda")
    return pipeline
    
def load_custom_diffusion_pipeline(model_path, unet_path, args):
    pipeline = DiffusionPipeline.from_pretrained(
        model_path,
        safety_checker=None,
        weight_dtype=torch.float32,
    ).to("cuda")
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    # load attention processors
    weight_name = (
        "pytorch_custom_diffusion_weights.safetensors"
        if not args.no_safe_serialization
        else "pytorch_custom_diffusion_weights.bin"
    )
    pipeline.unet.load_attn_procs(unet_path, weight_name=weight_name)

    modifier_token = args.concept_name.split("+")

    for token in modifier_token:
        token_weight_name = f"{token}.safetensors" if not args.no_safe_serialization else f"{token}.bin"
        pipeline.load_textual_inversion(unet_path, weight_name=token_weight_name)

    return pipeline

def load_my_pipeline(model_path, embedding_path, concept_name):
    assign_func_call(StableDiffusionPipeline)
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")

    placeholder_token = [f"{concept_name}"]
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    if num_added_tokens == 0:
        raise ValueError("The tokens are already in the tokenizer")
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    text_encoder.resize_token_embeddings(len(tokenizer))

    learned_embeds = torch.load(embedding_path)
    token_embeds = text_encoder.get_input_embeddings().weight.data
    for token, token_id in zip(placeholder_token, placeholder_token_id):
        token_embeds[token_id] = learned_embeds[token]

    pipeline = my_pipelines.StableDiffusionPipeline.from_pretrained(
        model_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        safety_checker=None,
        weight_dtype=torch.float32,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to("cuda")
    return pipeline

if __name__ == "__main__":
    args = parse_args()

    if args.method == "standard":
        main_standard(args)
    elif args.method == "ti":
        main_ti(args)
    elif args.method == "custom_diffusion":
        main_custom_diffusion(args)
    elif args.method == "gen_from_prompt":
        main_gen_from_prompt(args)
    else:
        raise ValueError(f"Unknown method: {args.method}")
