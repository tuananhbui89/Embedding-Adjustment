# Model Training Guide

This document provides a step-by-step guide for training the model in this project.

## Environment Setup

1. Ensure the following dependencies are installed:
   - Python 3.10.16
   - PyTorch 2.5.1+cu121
   - Required libraries (install via `requirements.txt`)

   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

- Ensure the data format matches the requirements of the training dataset (e.g., `examples/pose.jsonl`, `examples/subject.jsonl`, `examples/style.jsonl`).

## Start Training

1. Use the following commands to start training:

   - For spatial control:
     ```bash
     bash ./train_spatial.sh
     ```
   - For subject control:
     ```bash
     bash ./train_subject.sh
     ```
   - For style control:
     ```bash
     bash ./train_style.sh
     ```

2. Example training configuration:

   ```bash
   --pretrained_model_name_or_path $MODEL_DIR \  # Path to the FLUX model
   --cond_size=512 \  # Source image size (recommended: 384-512 or higher for better detail control)
   --noise_size=1024 \  # Target image's longest side size (recommended: 1024 for better resolution)
   --subject_column="None" \  # JSONL key for subject; set to "None" if using spatial condition
   --spatial_column="source" \  # JSONL key for spatial; set to "None" if using subject condition
   --target_column="target" \  # JSONL key for the target image
   --caption_column="caption" \  # JSONL key for the caption
   --ranks 128 \  # LoRA rank (recommended: 128)
   --network_alphas 128 \  # LoRA network alphas (recommended: 128)
   --output_dir=$OUTPUT_DIR \  # Directory for model and validation outputs
   --logging_dir=$LOG_PATH \  # Directory for logs
   --mixed_precision="bf16" \  # Recommended: bf16
   --train_data_dir=$TRAIN_DATA \  # Path to the training data JSONL file
   --learning_rate=1e-4 \  # Recommended: 1e-4
   --train_batch_size=1 \  # Only supports 1 due to multi-resolution target images
   --validation_prompt "Ghibli Studio style, Charming hand-drawn anime-style illustration" \  # Validation prompt
   --num_train_epochs=1000 \  # Total number of epochs
   --validation_steps=20 \  # Validate every n steps
   --checkpointing_steps=20 \  # Save model every n steps
   --spatial_test_images "./examples/style_data/5.png" \  # Validation images for spatial condition
   --subject_test_images None \  # Validation images for subject condition
   --test_h 1024 \  # Height of validation images
   --test_w 1024 \  # Width of validation images
   --num_validation_images=2  # Number of validation images
   ```

## Model Inference

1. After training, use the following script for inference:

   ```bash
   # Navigate to the original repository to use KV cache
   cd ..
   ```

   ```python
   import torch
   from PIL import Image
   from src.pipeline import FluxPipeline
   from src.transformer_flux import FluxTransformer2DModel
   from src.lora_helper import set_single_lora, set_multi_lora

   def clear_cache(transformer):
       for name, attn_processor in transformer.attn_processors.items():
           attn_processor.bank_kv.clear()

   # Initialize the model
   device = "cuda"
   base_path = "black-forest-labs/FLUX.1-dev"  # Path to the base model
   pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16, device=device)
   transformer = FluxTransformer2DModel.from_pretrained(
       base_path, 
       subfolder="transformer",
       torch_dtype=torch.bfloat16, 
       device=device
   )
   pipe.transformer = transformer
   pipe.to(device)

   # Path to your trained EasyControl model
   lora_path = "  "

   # Single condition control example
   set_single_lora(pipe.transformer, path, lora_weights=[1], cond_size=512)

   # Set your control image path
   spatial_image_path = ""
   subject_image_path = ""
   style_image_path = ""

   control_image = Image.open("fill in spatial_image_path or subject_image_path !!")
   prompt = "fill in your prompt!!"

   # For spatial or style control
   image = pipe(
       prompt,
       height=768,  # Generated image height
       width=1024,  # Generated image width
       guidance_scale=3.5,
       num_inference_steps=25,  # Number of steps
       max_sequence_length=512,
       generator=torch.Generator("cpu").manual_seed(5),
       spatial_images=[control_image],
       subject_images=[],
       cond_size=512,  # Training setting for cond_size
   ).images[0]
   # Clear cache after generation
   clear_cache(pipe.transformer)
   image.save("output.png")
   ```

2. For subject control:

   ```python
   image = pipe(
       prompt,
       height=768,
       width=1024,
       guidance_scale=3.5,
       num_inference_steps=25,
       max_sequence_length=512,
       generator=torch.Generator("cpu").manual_seed(5),
       spatial_images=[],
       subject_images=[control_image],
       cond_size=512,
   ).images[0]
   # Clear cache after generation
   clear_cache(pipe.transformer)
   image.save("output.png")
   ```

3. For multi-condition control:

   ```python
   import torch
   from PIL import Image
   from src.pipeline import FluxPipeline
   from src.transformer_flux import FluxTransformer2DModel
   from src.lora_helper import set_single_lora, set_multi_lora

   def clear_cache(transformer):
       for name, attn_processor in transformer.attn_processors.items():
           attn_processor.bank_kv.clear()

   # Initialize the model
   device = "cuda"
   base_path = "black-forest-labs/FLUX.1-dev"  # Path to the base model
   pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16, device=device)
   transformer = FluxTransformer2DModel.from_pretrained(
       base_path, 
       subfolder="transformer",
       torch_dtype=torch.bfloat16, 
       device=device
   )
   pipe.transformer = transformer
   pipe.to(device)

   # Change to your EasyControl Model path!!!
   lora_path = "./models"
   control_models = {
       "canny": f"{lora_path}/canny.safetensors",
       "depth": f"{lora_path}/depth.safetensors",
       "hedsketch": f"{lora_path}/hedsketch.safetensors",
       "pose": f"{lora_path}/pose.safetensors",
       "seg": f"{lora_path}/seg.safetensors",
       "inpainting": f"{lora_path}/inpainting.safetensors",
       "subject": f"{lora_path}/subject.safetensors",
   }
   paths = [control_models["subject"], control_models["inpainting"]]
   set_multi_lora(pipe.transformer, paths, lora_weights=[[1], [1]], cond_size=512)

   # Subject + spatial control
   prompt = "A SKS on the car"
   subject_images = [Image.open("./test_imgs/subject_1.png").convert("RGB")]
   spatial_images = [Image.open("./test_imgs/inpainting.png").convert("RGB")]
   image = pipe(
       prompt,
       height=1024,
       width=1024,
       guidance_scale=3.5,
       num_inference_steps=25,
       max_sequence_length=512,
       generator=torch.Generator("cpu").manual_seed(42),
       subject_images=subject_images,
       spatial_images=spatial_images,
       cond_size=512,
   ).images[0]
   # Clear cache after generation
   clear_cache(pipe.transformer)
   image.save("output_multi.png")
   ```

4. For spatial + spatial control:

   ```python
   prompt = "A car"
   subject_images = []
   spatial_images = [Image.open("image1_path").convert("RGB"), Image.open("image2_path").convert("RGB")]
   image = pipe(
       prompt,
       height=1024,
       width=1024,
       guidance_scale=3.5,
       num_inference_steps=25,
       max_sequence_length=512,
       generator=torch.Generator("cpu").manual_seed(42),
       subject_images=subject_images,
       spatial_images=spatial_images,
       cond_size=512,
   ).images[0]
   # Clear cache after generation
   clear_cache(pipe.transformer)
   image.save("output_multi.png")
   ```

## Notes

- Adjust `noise_size` and `cond_size` based on your VRAM.
- Batch size is limited to 1.
