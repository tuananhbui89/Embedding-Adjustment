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

import my_pipelines

from utils import set_seed, read_prompt_file
from algo_adjust_embedding import adjust_norm_and_slerp


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--concept_name", type=str, default="<cathy>")
    parser.add_argument("--concept_name_2", type=str, default="<dog>")
    parser.add_argument("--output_dir", type=str, default="evaluation_folder")
    parser.add_argument("--model_path", type=str, default="outputs/cathy_4")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--projector_path", type=str, default="outputs/cathy_4/projector.pth")
    parser.add_argument("--method", type=str, default="gti")
    parser.add_argument("--projector_type", type=str, default="simple_projector")
    parser.add_argument("--embedding_path", type=str, default="outputs/cathy_4/learned_embeds.bin")
    parser.add_argument("--embedding_path_2", type=str, default="outputs/celebA_ti_432/learned_embeds.bin")
    parser.add_argument("--max_train_steps", type=int, default=3000)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--prompt_file", type=str, default="None")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--start_steps", type=int, default=0)
    parser.add_argument("--unet_path", type=str, default="None")
    parser.add_argument("--start_prompt_index", type=int, default=0)
    parser.add_argument("--target_word", type=str, default="man")
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument("--target_word_2", type=str, default="dog")
    parser.add_argument("--rho_2", type=float, default=0.2)
    parser.add_argument("--alpha_2", type=float, default=0.5)

    args = parser.parse_args()
    set_seed(args.seed)
    return args


def main_adjust_embedding(args):
    generator = torch.Generator("cuda").manual_seed(args.seed)

    pipeline = load_pipeline_two_objects(args.model_path, args.embedding_path, args.embedding_path_2, args.concept_name, args.concept_name_2)
    # create output directory
    prompt_name = args.prompt_file.split("/")[-1].split(".")[0]
    output_dir = f"{args.output_dir}/combine_{args.concept_name}_{args.concept_name_2}/{prompt_name}"
    os.makedirs(output_dir, exist_ok=True)

    assert os.path.exists(args.prompt_file), f"Prompt file {args.prompt_file} does not exist"
    prompts = read_prompt_file(args.prompt_file)

    assert "{object1}" in prompts[0] and "{object2}" in prompts[0], f"Prompt {prompts[0]} does not contain {{object1}} and {{object2}}"

    if args.start_steps == 0:
        for i, prompt in enumerate(prompts):
            if i < args.start_prompt_index:
                continue
            for j in range(args.num_images):
                assert "{object1}" in prompt and "{object2}" in prompt, f"Prompt {prompt} does not contain {{object1}} and {{object2}}"
                print(i, j, "prompt: ", prompt.replace("{object1}", args.concept_name).replace("{object2}", args.concept_name_2))
                image = pipeline(prompt.replace("{object1}", args.concept_name).replace("{object2}", args.concept_name_2), num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, generator=generator).images[0]
                image.save(f"{output_dir}/{i}_{j}.png")


def load_pipeline_two_objects(model_path, embedding_path, embedding_path_2, concept_name, concept_name_2):
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")

    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")


    if ' ' in concept_name:
        print(f"Concept name {concept_name} contains a space, splitting it into two tokens")
        placeholder_token = concept_name.split(' ')
    else:
        placeholder_token = [f"{concept_name}"]

    if ' ' in concept_name_2:
        print(f"Concept name {concept_name_2} contains a space, splitting it into two tokens")
        placeholder_token_2 = concept_name_2.split(' ')
    else:
        placeholder_token_2 = [f"{concept_name_2}"]

    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    assert num_added_tokens == len(placeholder_token), f"Number of added tokens {num_added_tokens} does not match the number of placeholder tokens {len(placeholder_token)}"

    num_added_tokens_2 = tokenizer.add_tokens(placeholder_token_2)
    assert num_added_tokens_2 == len(placeholder_token_2), f"Number of added tokens {num_added_tokens_2} does not match the number of placeholder tokens {len(placeholder_token_2)}"

    if num_added_tokens == 0:
        raise ValueError("The tokens are already in the tokenizer")
    if num_added_tokens_2 == 0:
        raise ValueError("The tokens are already in the tokenizer")

    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    assert len(placeholder_token_id) == len(placeholder_token), f"Number of placeholder token ids {len(placeholder_token_id)} does not match the number of placeholder tokens {len(placeholder_token)}"

    placeholder_token_id_2 = tokenizer.convert_tokens_to_ids(placeholder_token_2)
    assert len(placeholder_token_id_2) == len(placeholder_token_2), f"Number of placeholder token ids {len(placeholder_token_id_2)} does not match the number of placeholder tokens {len(placeholder_token_2)}"
   
    text_encoder.resize_token_embeddings(len(tokenizer))

    learned_embeds = torch.load(embedding_path)

    learned_embeds_2 = torch.load(embedding_path_2)
    token_embeds = text_encoder.get_input_embeddings().weight.data

    print(f"learned_embeds len: {len(learned_embeds)}")
    print(f"token_embeds len: {len(token_embeds)}")

    # Our approach description here 
    # Goal: Adjust the norm of the learned embeddings within a range, given by
    tokenizer_vocab = tokenizer.get_vocab()

    assert type(args.target_word) == str, f"target_word should be a tensor or a string"
    assert args.target_word in tokenizer_vocab, f"target_word {args.target_word} is not in the tokenizer vocabulary"
    target_token_id = tokenizer.convert_tokens_to_ids(args.target_word)
    target_vector = token_embeds[target_token_id]

    assert type(args.target_word_2) == str, f"target_word_2 should be a tensor or a string"
    assert args.target_word_2 in tokenizer_vocab, f"target_word_2 {args.target_word_2} is not in the tokenizer vocabulary"
    target_token_id_2 = tokenizer.convert_tokens_to_ids(args.target_word_2)
    target_vector_2 = token_embeds[target_token_id_2]

    for i in range(len(placeholder_token_id)):
        # NOTE HERE: We save the embeddings for the two placeholder tokens, but using args.placeholder_token as the key
        token_embeds[placeholder_token_id[i]] = adjust_norm_and_slerp(learned_embeds[placeholder_token[0]][i], target_vector, args.rho, args.alpha)
    
    for i in range(len(placeholder_token_id_2)):
        token_embeds[placeholder_token_id_2[i]] = adjust_norm_and_slerp(learned_embeds_2[placeholder_token_2[0]][i], target_vector_2, args.rho_2, args.alpha_2)

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

