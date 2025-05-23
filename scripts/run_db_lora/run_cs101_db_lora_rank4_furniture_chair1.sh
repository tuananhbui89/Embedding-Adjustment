#!/bin/bash

CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 --main_process_port=29503 train_dreambooth_lora.py \
    --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5"  \
    --instance_data_dir="cs101/furniture_chair1" \
    --class_data_dir="evaluation_folder/prior_images/chair" \
    --output_dir="outputs/cs101_db_lora_rank4_furniture_chair1" \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="sks chair" \
    --class_prompt="a photo of a chair" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=200 \
    --max_train_steps=3000 \
    --checkpointing_steps=50 \
    --validation_prompt="a photo of a sks chair wearing glasses and writing on a red notebook" \
    --validation_steps=50 \
    --rank=4 \
    --train_text_encoder

CUDA_VISIBLE_DEVICES=1 python generate_dreambooth.py \
    --model_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --lora_path="outputs/cs101_db_lora_rank4_furniture_chair1/checkpoint-2000" \
    --output_dir="evaluation_massive/cs101_db_lora_rank4_furniture_chair1/2000" \
    --prompt_file="prompts/gen_prompt_cs101_chair.csv" \
    --concept_name="sks chair" \
    --num_images=50 \
    --num_inference_steps=50 \
    --seed=42 \
    --train_text_encoder \
    --guidance_scale=7.0

for custom_prompt in objectA objectB full_prompt; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
        --images_folder="evaluation_massive/cs101_db_lora_rank4_furniture_chair1/2000/sks chair/gen_prompt_cs101_chair" \
        --prompt_file="prompts/gen_prompt_cs101_chair.csv" \
        --num_images=50 \
        --use_custom_prompt=$custom_prompt \
        --info="use_custom_prompt_$custom_prompt" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_db_lora_rank4_furniture_chair1/2000"
done

CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
    --images_folder="evaluation_massive/cs101_db_lora_rank4_furniture_chair1/2000/sks chair/gen_prompt_cs101_chair" \
    --prompt_file="prompts/gen_prompt_cs101_chair.csv" \
    --num_images=50 \
    --anchor_image_path="cs101/furniture_chair1/0.jpeg" \
    --info="t01" \
    --sub_folder="None" \
    --output_dir="semantic_drift/cs101_db_lora_rank4_furniture_chair1/2000"

CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
    --images_folder="evaluation_massive/cs101_db_lora_rank4_furniture_chair1/2000/sks chair/gen_prompt_cs101_chair" \
    --prompt_file="prompts/gen_prompt_cs101_chair.csv" \
    --num_images=50 \
    --anchor_image_path="cs101/furniture_chair1/0.jpeg" \
    --info="t01" \
    --sub_folder="None" \
    --output_dir="semantic_drift/cs101_db_lora_rank4_furniture_chair1/2000"
