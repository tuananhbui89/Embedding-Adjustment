#!/bin/bash

for RHO in 0.1; do
    CUDA_VISIBLE_DEVICES=1 python generate_db_adjust_embedding_v4.py \
        --model_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
        --lora_path="outputs/cs101_db_lora_rank4_scene_barn/checkpoint-2000" \
        --output_dir="evaluation_massive/cs101_db_lora_rank4_scene_barn_adjust_embedding_v4/rho${RHO}_alpha0.5/2000" \
        --prompt_file="prompts/gen_prompt_cs101_barn.csv" \
        --concept_name="sks barn" \
        --target_name="barn" \
        --num_images=50 \
        --num_inference_steps=50 \
        --seed=42 \
        --train_text_encoder \
        --guidance_scale=7.0 \
        --rho=$RHO \
        --alpha=0.5
done

for RHO in 0.1; do
  for custom_prompt in objectA objectB full_prompt; do
    CUDA_VISIBLE_DEVICES=1 python investigate_clip_sim_v2.py \
        --images_folder="evaluation_massive/cs101_db_lora_rank4_scene_barn_adjust_embedding_v4/rho${RHO}_alpha0.5/2000/sks barn/gen_prompt_cs101_barn" \
        --prompt_file="prompts/gen_prompt_cs101_barn.csv" \
        --num_images=50 \
        --use_custom_prompt=$custom_prompt \
        --info="use_custom_prompt_$custom_prompt" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_db_lora_rank4_scene_barn_adjust_embedding_v4/rho${RHO}_alpha0.5/2000"
  done
done

for RHO in 0.1; do
    CUDA_VISIBLE_DEVICES=1 python investigate_clip_sim_image.py \
        --images_folder="evaluation_massive/cs101_db_lora_rank4_scene_barn_adjust_embedding_v4/rho${RHO}_alpha0.5/2000/sks barn/gen_prompt_cs101_barn" \
        --prompt_file="prompts/gen_prompt_cs101_barn.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/scene_barn/2.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_db_lora_rank4_scene_barn_adjust_embedding_v4/rho${RHO}_alpha0.5/2000"
done

for RHO in 0.1; do
    CUDA_VISIBLE_DEVICES=1 python investigate_dino_sim_image.py \
        --images_folder="evaluation_massive/cs101_db_lora_rank4_scene_barn_adjust_embedding_v4/rho${RHO}_alpha0.5/2000/sks barn/gen_prompt_cs101_barn" \
        --prompt_file="prompts/gen_prompt_cs101_barn.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/scene_barn/2.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_db_lora_rank4_scene_barn_adjust_embedding_v4/rho${RHO}_alpha0.5/2000"
done
