import torch
import argparse
import os
import numpy as np
import random

from diffusers import StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import DiffusionPipeline
from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel

from transformers import CLIPTextModel, CLIPTokenizer

from utils import set_seed, read_prompt_file
from algo_adjust_embedding import adjust_norm_and_slerp

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
    parser.add_argument("--max_train_steps", type=int, default=3000)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--prompt_file", type=str, default="None")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--start_steps", type=int, default=0)
    parser.add_argument("--target_word", type=str, default="woman")
    parser.add_argument("--rho", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--start_prompt_index", type=int, default=0)

    args = parser.parse_args()
    set_seed(args.seed)
    return args

def main_adjust_embedding(args):
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
                assert "{}" in prompt, f"Prompt {prompt} does not contain {{}}"
                print(i, j, "prompt: ", prompt.format(args.concept_name))
                image = pipeline(prompt=prompt.format(args.concept_name), num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, generator=generator).images[0]
                image.save(f"{output_dir}/{steps}/{i}_{j}.png")
       

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

    # for i in range(len(placeholder_token_id)):
    #     # NOTE HERE: We save the embeddings for the two placeholder tokens, but using args.placeholder_token as the key
    #     token_embeds[placeholder_token_id[i]] = learned_embeds[placeholder_token[0]][i]

    # Our approach description here 
    # Goal: Adjust the norm of the learned embeddings within a range, given by
    tokenizer_vocab = tokenizer.get_vocab()

    assert type(args.target_word) == str, f"target_word should be a tensor or a string"
    assert args.target_word in tokenizer_vocab, f"target_word {args.target_word} is not in the tokenizer vocabulary"
    target_token_id = tokenizer.convert_tokens_to_ids(args.target_word)
    target_vector = token_embeds[target_token_id]


    for i in range(len(placeholder_token_id)):
        token_embeds[placeholder_token_id[i]] = adjust_norm_and_slerp(learned_embeds[placeholder_token[0]][i], target_vector, args.rho, args.alpha)


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

if __name__ == "__main__":
    args = parse_args()

    main_adjust_embedding(args)
