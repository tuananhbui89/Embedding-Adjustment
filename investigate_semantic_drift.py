from diffusers import StableDiffusionPipeline
import torch
import argparse
import os
from generate_textual_inversion import load_pipeline

import my_pipelines
from diffusers import DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from my_pipelines import SimpleProjector, SimpleProjectorOneLayer
from my_pipelines import assign_func_call
from my_pipelines import StableDiffusionPipeline

from utils import read_prompt_file
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

    parser.add_argument("--prompt_file", type=str, default="prompt_cat_toy.csv")
    parser.add_argument("--initializer_token", type=str, default="toy")
    parser.add_argument("--placeholder_token", type=str, default="<celebA>")


    return parser.parse_args()


@torch.no_grad()
def main(args):
    # load prompt file
    prompts = read_prompt_file(args.prompt_file)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder")

    placeholder_token = [f"{args.placeholder_token}"]
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    if num_added_tokens == 0:
        raise ValueError("The tokens are already in the tokenizer")
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    text_encoder.resize_token_embeddings(len(tokenizer))

    text_encoder.to("cuda")

    if os.path.exists(args.embedding_path):
        learned_embeds = torch.load(args.embedding_path)
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for token, token_id in zip(placeholder_token, placeholder_token_id):
            token_embeds[token_id] = learned_embeds[token]

        # generate images
        reference_embeddings = []
        for prompt in prompts:

            tokenized_prompt = tokenizer(
                prompt,
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
        save_folder = os.path.join(args.output_dir, args.prompt_file.replace('.csv', ''))
        os.makedirs(save_folder, exist_ok=True)
        torch.save(reference_embeddings, os.path.join(save_folder, "reference_embeddings.pt"))

    dict_embeddings = {}

    
    for i in range(5000):
        embedding_path = args.embedding_path.replace(".bin", f"-steps-{i}.bin")
        if not os.path.exists(embedding_path):
            continue
        print("--------------------------------")
        print(f"Loading embedding from {embedding_path}")

        learned_embeds = torch.load(embedding_path)
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for token, token_id in zip(placeholder_token, placeholder_token_id):
            token_embeds[token_id] = learned_embeds[token]

        new_embeddings = []
        for prompt in prompts:
            # sanity check - only one instance of the initializer token
            assert args.initializer_token in prompt
            assert prompt.count(args.initializer_token) == 1
            _prompt = prompt.strip()
            _prompt = _prompt.replace(args.initializer_token, args.concept_name)
            print(f"Prompt: {_prompt}")

            tokenized_prompt = tokenizer(
                _prompt,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",).input_ids.to("cuda")  # Move to device
            
            # Get text embeddings, keeping batch dimension
            # Get the entire sequence of hidden states of the text encoder
            embedding = text_encoder(tokenized_prompt)[0]
            new_embeddings.append(embedding)

        new_embeddings = torch.cat(new_embeddings, dim=0)
        dict_embeddings[i] = new_embeddings

        # clear memory
        torch.cuda.empty_cache()

    save_folder = os.path.join(args.output_dir, args.prompt_file.replace('.csv', ''))
    os.makedirs(save_folder, exist_ok=True)
    torch.save(dict_embeddings, os.path.join(save_folder, "dict_embeddings.pt"))
    
if __name__ == "__main__":
    args = parse_args()
    main(args)