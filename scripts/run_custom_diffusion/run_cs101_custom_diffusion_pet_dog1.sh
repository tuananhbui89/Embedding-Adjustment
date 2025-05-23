#!/bin/bash

CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes=1 --main_process_port=29501 train_custom_diffusion_and_eval.py \
    --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --instance_data_dir="cs101/pet_dog1" \
    --output_dir="outputs/cs101_custom_diffusion_pet_dog1" \
    --class_data_dir="evaluation_folder/prior_images/dog" \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --class_prompt="a photo of a dog" --num_class_images=200 \
    --instance_prompt="photo of a <dog> dog" \
    --resolution=512 \
    --train_batch_size=1 \
    --learning_rate=5e-6 \
    --lr_warmup_steps=0 \
    --max_train_steps=2000 \
    --scale_lr --hflip --noaug \
    --freeze_model crossattn \
    --modifier_token "<dog>" \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=100 \
    --validation_prompt="photo of a <dog> dog wearing glasses and writing on a red notebook" \
    --validation_steps=50 \
    --no_safe_serialization\
    --eval_steps="1000"\
    --target_name="dog"\
    --rho=0.2\
    --alpha=0.5\
    --output_images_dir="evaluation_massive/cs101_custom_diffusion_pet_dog1"\
    --output_images_dir_ea="evaluation_massive/cs101_custom_diffusion_pet_dog1_adjust_embedding_v4/rho0.2_alpha0.5"\
    --prompt_file="prompts/gen_prompt_cs101_dog.csv"\
    --num_inference_steps=50\
    --guidance_scale=7.0\
    --num_images=50


for custom_prompt in objectA objectB full_prompt; do
    CUDA_VISIBLE_DEVICES=2 python investigate_clip_sim_v2.py \
        --images_folder="evaluation_massive/cs101_custom_diffusion_pet_dog1/<dog>/gen_prompt_cs101_dog/1000" \
        --prompt_file="prompts/gen_prompt_cs101_dog.csv" \
        --num_images=50 \
        --use_custom_prompt=$custom_prompt \
        --info="use_custom_prompt_$custom_prompt" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_custom_diffusion_pet_dog1/1000"
done

for custom_prompt in objectA objectB full_prompt; do
    CUDA_VISIBLE_DEVICES=2 python investigate_clip_sim_v2.py \
        --images_folder="evaluation_massive/cs101_custom_diffusion_pet_dog1_adjust_embedding_v4/rho0.2_alpha0.5/<dog>/gen_prompt_cs101_dog/1000" \
        --prompt_file="prompts/gen_prompt_cs101_dog.csv" \
        --num_images=50 \
        --use_custom_prompt=$custom_prompt \
        --info="use_custom_prompt_$custom_prompt" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_custom_diffusion_pet_dog1_adjust_embedding_v4/rho0.2_alpha0.5/1000"
done

CUDA_VISIBLE_DEVICES=2 python investigate_clip_sim_image.py \
        --images_folder="evaluation_massive/cs101_custom_diffusion_pet_dog1/<dog>/gen_prompt_cs101_dog/1000" \
        --prompt_file="prompts/gen_prompt_cs101_dog.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/pet_dog1/6.jpeg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_custom_diffusion_pet_dog1/1000"

CUDA_VISIBLE_DEVICES=2 python investigate_clip_sim_image.py \
        --images_folder="evaluation_massive/cs101_custom_diffusion_pet_dog1_adjust_embedding_v4/rho0.2_alpha0.5/<dog>/gen_prompt_cs101_dog/1000" \
        --prompt_file="prompts/gen_prompt_cs101_dog.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/pet_dog1/6.jpeg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_custom_diffusion_pet_dog1_adjust_embedding_v4/rho0.2_alpha0.5/1000"
CUDA_VISIBLE_DEVICES=2 python investigate_dino_sim_image.py \
        --images_folder="evaluation_massive/cs101_custom_diffusion_pet_dog1/<dog>/gen_prompt_cs101_dog/1000" \
        --prompt_file="prompts/gen_prompt_cs101_dog.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/pet_dog1/6.jpeg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_custom_diffusion_pet_dog1/1000"

CUDA_VISIBLE_DEVICES=2 python investigate_dino_sim_image.py \
        --images_folder="evaluation_massive/cs101_custom_diffusion_pet_dog1_adjust_embedding_v4/rho0.2_alpha0.5/<dog>/gen_prompt_cs101_dog/1000" \
        --prompt_file="prompts/gen_prompt_cs101_dog.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/pet_dog1/6.jpeg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_custom_diffusion_pet_dog1_adjust_embedding_v4/rho0.2_alpha0.5/1000"
