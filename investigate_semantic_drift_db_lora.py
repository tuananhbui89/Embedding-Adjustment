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
    parser.add_argument("--placeholder_token", type=str, default="sks")
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )

    args = parser.parse_args()
    set_seed(args.seed)
    return args

def main():
    args = parse_args()
    print(args)

    generator = torch.Generator("cuda").manual_seed(args.seed)

    # create output directory
    prompt_name = args.prompt_file.split("/")[-1].split(".")[0]
    # output_dir = f"{args.output_dir}/{args.concept_name}/{prompt_name}"
    # os.makedirs(output_dir, exist_ok=True)

    assert os.path.exists(args.prompt_file), f"Prompt file {args.prompt_file} does not exist"
    prompts = read_prompt_file(args.prompt_file)

    unet = UNet2DConditionModel.from_pretrained(
        args.model_path, subfolder="unet"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_path, subfolder="text_encoder"
    )
    tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")


    pipeline = DiffusionPipeline.from_pretrained(args.model_path, unet=unet, text_encoder=text_encoder)
    pipeline.to("cuda")

    # Disable NSFW detector
    pipeline.safety_checker = None

    # get embeddings of prompts
    reference_embeddings = []
    for prompt in prompts:
        print(prompt.format(args.concept_name))
        tokenized_prompt = tokenizer(
            prompt.format(args.concept_name),
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to("cuda")  # Move to device

        # Get text embeddings, keeping batch dimension
        # Get the entire sequence of hidden states of the text encoder
        embedding = text_encoder(tokenized_prompt)[0]
        reference_embeddings.append(embedding)

    reference_embeddings = torch.cat(reference_embeddings, dim=0)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(reference_embeddings, os.path.join(args.output_dir, "reference_embeddings.pt"))

    # load lora weights
    print(f"Loading lora weights from {args.lora_path}")

    if args.train_text_encoder:
        pipeline.load_lora_weights(args.lora_path, weight_name="pytorch_lora_weights.safetensors")
    else:
        # load lora weights but specific to the unet - that works
        pipeline.unet.load_lora_adapter(args.lora_path, weight_name="pytorch_lora_weights.safetensors", prefix="unet")


    token_embeds = pipeline.text_encoder.get_input_embeddings().weight.data
    token_vocab = tokenizer.get_vocab()
    print('token_vocab', len(token_vocab.keys()))
    # print('token_vocab', token_vocab)

    # get embeddings of specific tokens
    # get the index of the placeholder token from token vocab. 
    # Important: Do not use convert_tokens_to_ids. It will return the index of the last token in the vocab.
    if args.placeholder_token not in token_vocab:
        print(f"Placeholder token {args.placeholder_token} not found in tokenizer")
        print('Adding special character to the placeholder token')
        investigate_placeholder_token = args.placeholder_token + '</w>'
    else:
        investigate_placeholder_token = args.placeholder_token

    if investigate_placeholder_token not in token_vocab:
        print(f"Placeholder token {investigate_placeholder_token} not found in tokenizer")
        raise ValueError("Placeholder token not found in tokenizer")

    placeholder_token_id = token_vocab[investigate_placeholder_token]
    specific_token_embeds = token_embeds[placeholder_token_id]

    print(specific_token_embeds.shape)
    # print(specific_token_embeds)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(specific_token_embeds.cpu(), os.path.join(args.output_dir, "specific_token_embeds.pt"))

    # Results: No change in the embeddings of the placeholder token. It might be because the embeddings are not updated.

    # get embeddings of prompts
    prompt_embeddings = []
    for prompt in prompts:
        print(prompt.format(args.concept_name))
        tokenized_prompt = tokenizer(
            prompt.format(args.concept_name),
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to("cuda")  # Move to device

        # Get text embeddings, keeping batch dimension
        # Get the entire sequence of hidden states of the text encoder
        embedding = text_encoder(tokenized_prompt)[0]
        prompt_embeddings.append(embedding)

    prompt_embeddings = torch.cat(prompt_embeddings, dim=0)
    torch.save(prompt_embeddings, os.path.join(args.output_dir, "prompt_embeddings.pt"))

if __name__ == "__main__":
    main()

