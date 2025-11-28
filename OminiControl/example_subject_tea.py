import os

import torch
from diffusers.pipelines import FluxPipeline
from diffusers import FluxTransformer2DModel
from PIL import Image
from transformers import T5EncoderModel

from omini.pipeline.flux_omini_with_tea import Condition, generate, seed_everything
import gc

import argparse
import utils
import time

parser = argparse.ArgumentParser()
parser.add_argument("--setting", type=str, default="tshirt")
parser.add_argument("--num_images", type=int, default=1)
parser.add_argument("--rho", type=float, default=0.2)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--prompt_index", type=int, default=None)
args = parser.parse_args()

setting = utils.settings[args.setting]

output_dir = f"evaluation_massive/omini_subject_tea/{args.setting}_{args.rho}_{args.alpha}"
os.makedirs(output_dir, exist_ok=True)

# Set CUDA memory allocation configuration to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

flush()

# Set custom Hugging Face cache directory
custom_transformer_dir = "/path/to/your/huggingface/cache/"
custom_cache_dir= "/path/to/your/huggingface/cache/"

os.environ["HF_HOME"] = custom_cache_dir
os.environ["TRANSFORMERS_CACHE"] = custom_transformer_dir
os.makedirs(custom_cache_dir, exist_ok=True)
os.makedirs(custom_transformer_dir, exist_ok=True)

# Load T5 Text Encoder in 4-Bit mode with consistent dtype
ckpt_id = "black-forest-labs/FLUX.1-schnell"

pipe = FluxPipeline.from_pretrained(
    ckpt_id, torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")

pipe.load_lora_weights(
    "Yuanshi/OminiControl",
    weight_name=f"omini/subject_512.safetensors",
    adapter_name="subject",
)

image = Image.open(setting["image"]).convert("RGB").resize((512, 512))

# For this model, the position_delta is (0, 32).
# For more details of position_delta, please refer to:
# https://github.com/Yuanshi9815/OminiControl/issues/89#issuecomment-2827080344
condition = Condition(image, "subject", position_delta=(0, 32))


keyword = setting["keyword"]
target_keyword = setting["target_keyword"]
template_prompts = setting["template_prompts"]

prompts = [prompt.format(keyword) for prompt in template_prompts]
target_prompts = [prompt.format(target_keyword) for prompt in template_prompts]

seed_everything(0)
start_time = time.time()
for idx, prompt in enumerate(prompts):
    if args.prompt_index is not None and idx != args.prompt_index:
        continue
    for n_idx in range(args.num_images):
        current_time = time.time()
        print(idx, n_idx, prompt, target_prompts[idx], f"Time taken: {current_time - start_time} seconds")
        result_img = generate(
            pipe,
            target_prompt=target_prompts[idx],
            rho=args.rho,
            alpha=args.alpha,
            prompt=prompt,
            conditions=[condition],
            num_inference_steps=8,
            height=512,
            width=512,
        ).images[0]

        result_img.save(f"{output_dir}/{idx}_{n_idx}.png")