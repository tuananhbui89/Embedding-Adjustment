import torch
from PIL import Image
from src.pipeline_with_tea import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora, load_checkpoint
from peft import LoraConfig
import os
from find_latest_checkpoint import find_latest_checkpoint
import argparse

def read_prompt_file(prompt_file_path):
    assert os.path.exists(prompt_file_path), f"Prompt file {prompt_file_path} does not exist"
    assert prompt_file_path.endswith(".csv"), f"Prompt file {prompt_file_path} is not a csv file"

    # if there is a header, read from the prompt column only
    with open(prompt_file_path, "r") as f:
        prompts = f.readlines()
    if prompts[0].startswith("prompt"):
        prompts = [line.split(",")[0] for line in prompts[1:]]
    else:
        prompts = [line.strip() for line in prompts]
    
    # assert '{}' in prompts[0], f"Prompt file {prompt_file} does not contain {{}}"

    for i, prompt in enumerate(prompts):
        print('read_prompt_file', i, prompt)

    return prompts

def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

def load_text_encoder_lora(text_encoder, lora_path, device="cuda"):
    """Load LoRA weights for text encoder using PEFT"""
    if not os.path.exists(lora_path):
        print(f"Warning: Text encoder LoRA not found at {lora_path}")
        return text_encoder
        
    print(f"Loading text encoder LoRA from {lora_path}")
    
    # Create LoRA config matching the training configuration
    text_lora_config = LoraConfig(
        r=32,  # text_encoder_rank from training script
        lora_alpha=32,  # text_encoder_alpha from training script
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.0,
        init_lora_weights="gaussian",
    )
    
    # Add adapter to text encoder
    text_encoder.add_adapter(text_lora_config)
    
    # Load the LoRA weights
    lora_state_dict = load_checkpoint(lora_path)
    
    # Filter and load only LoRA weights
    lora_weights = {k: v for k, v in lora_state_dict.items() if 'lora' in k}
    
    # Load the weights into the model
    text_encoder.load_state_dict(lora_weights, strict=False)
    text_encoder.to(device)
    
    return text_encoder

def main(args):
    # Automatically find the latest checkpoint if the specified one doesn't exist or if None
    if args.checkpoint_step is None:
        args.checkpoint_step = find_latest_checkpoint(args.trained_model_dir)
        if args.checkpoint_step is None:
            raise ValueError(f"No checkpoints found in {args.trained_model_dir}")
        print(f"Using latest checkpoint: {args.checkpoint_step}")

    checkpoint_path = os.path.join(args.trained_model_dir, args.checkpoint_step)

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found at {checkpoint_path}")

    # Initialize model
    device = "cuda"
    # base_path = "black-forest-labs/FLUX.1-dev"  # Path to your base model
    base_path = "./models/flux-1-dev"

    # Load trained LoRA models from checkpoint
    transformer_lora_path = os.path.join(checkpoint_path, "transformer_lora.safetensors")
    text_encoder_lora_path = os.path.join(checkpoint_path, "text_encoder_one_lora.safetensors")

    # Load pipeline using the simple approach that works
    print("Loading base pipeline...")
    pipeline = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16, device=device)
    
    # Load transformer separately to apply LoRA
    transformer = FluxTransformer2DModel.from_pretrained(
        base_path, 
        subfolder="transformer",
        torch_dtype=torch.bfloat16, 
        device=device)
    
    # Replace the transformer in the pipeline
    pipeline.transformer = transformer
    pipeline.to(device)

    # Load transformer LoRA
    print(f"Loading transformer LoRA from {transformer_lora_path}")
    set_single_lora(pipeline.transformer, transformer_lora_path, lora_weights=[1], cond_size=args.cond_size) 
    
    # Load text encoder LoRA
    print(f"Loading text encoder LoRA from {text_encoder_lora_path}")
    pipeline.text_encoder = load_text_encoder_lora(pipeline.text_encoder, text_encoder_lora_path, device)

    # For subject generation, you might want to provide subject images
    subject_image_path = args.subject_image_path  # From training script
    if os.path.exists(subject_image_path):
        subject_image = Image.open(subject_image_path)
        subject_images = [subject_image]
    else:
        raise ValueError(f"Subject image not found at {subject_image_path}")

    prompts = read_prompt_file(args.prompt_file_path)

    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed else None

    for i, prompt in enumerate(prompts):
        pipeline_args = {"prompt": prompt.format(args.subject_token),
                        "target_prompt": prompt.format(args.target_token),
                        "spatial_images": [],
                        "subject_images": subject_images,
                        "height": args.height,
                        "width": args.width,
                        "cond_size": args.cond_size,
                        "guidance_scale": 3,
                        "num_inference_steps": 20,
                        "max_sequence_length": 512,
                        "rho": args.rho,
                        "alpha": args.alpha,
                        }

        print(f"Generating image...{prompt.format(args.subject_token)} with target prompt {prompt.format(args.target_token)}")
        images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_images)]

        # Save the generated image
        os.makedirs(args.output_path, exist_ok=True)
        for im, image in enumerate(images):
            image.save(f"{args.output_path}/{i}_{im}.png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_step", type=str, default=None, help="Checkpoint to use (e.g., 'checkpoint-50') or None for latest")
    parser.add_argument("--trained_model_dir", type=str, default="./models/subject_model_with_text_encoder", help="Directory containing trained model checkpoints")
    parser.add_argument("--subject_image_path", type=str, default="./examples/subject_data/3.png", help="Path to subject image")
    parser.add_argument("--prompt_file_path", type=str, default="./examples/prompt_data/prompt_data.csv", help="Path to prompt file")
    parser.add_argument("--height", type=int, default=768, help="Image height")
    parser.add_argument("--width", type=int, default=768, help="Image width")
    parser.add_argument("--cond_size", type=int, default=384, help="Conditioning size (should match training)")
    parser.add_argument("--output_path", type=str, default="output.png", help="Output image path")
    parser.add_argument("--subject_token", type=str, default="SKS", help="Subject token")
    parser.add_argument("--target_token", type=str, default="man", help="Target token")
    parser.add_argument("--rho", type=float, default=0.2, help="rho for TEA")
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha for TEA")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)