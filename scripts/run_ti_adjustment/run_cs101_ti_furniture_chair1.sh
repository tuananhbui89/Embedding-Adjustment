#!/bin/bash

for alpha in 0.5; do
    for rho in 0.2; do
        CUDA_VISIBLE_DEVICES=0 python generate_ti_adjust_embedding_v3.py \
            --method="standard" \
            --model_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
            --concept_name="<chair>" \
            --output_dir="evaluation_massive/cs101_ti_furniture_chair1_adjust_embedding_v3/rho${rho}_alpha${alpha}/1000" \
            --guidance_scale=7.0 \
            --embedding_path="outputs/cs101_ti_furniture_chair1/learned_embeds-steps-1000.bin" \
            --max_train_steps=3000 \
            --num_images=50 \
            --start_prompt_index=0 \
            --prompt_file="prompts/gen_prompt_cs101_chair.csv" \
            --target_word="chair" \
            --rho=$rho \
            --alpha=$alpha
    done
done

for alpha in 0.5; do
    for rho in 0.2; do
        for custom_prompt in 'objectA' 'objectB' 'full_prompt'; do
                CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
                --images_folder="evaluation_massive/cs101_ti_furniture_chair1_adjust_embedding_v3/rho${rho}_alpha0.5/1000/<chair>/gen_prompt_cs101_chair" \
                --prompt_file="prompts/gen_prompt_cs101_chair.csv" \
                --num_images=50 \
                --use_custom_prompt=$custom_prompt \
                --info="use_custom_prompt_${custom_prompt}" \
                --sub_folder="None" \
                --output_dir="semantic_drift/cs101_ti_furniture_chair1_adjust_embedding_v3/rho${rho}_alpha${alpha}/1000"
        done
    done
done

for alpha in 0.5; do
    for rho in 0.2; do
        CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
            --images_folder="evaluation_massive/cs101_ti_furniture_chair1_adjust_embedding_v3/rho${rho}_alpha0.5/1000/<chair>/gen_prompt_cs101_chair" \
            --prompt_file="prompts/gen_prompt_cs101_chair.csv" \
            --num_images=50 \
            --anchor_image_path="cs101/furniture_chair1/0.jpeg" \
            --info="t01" \
            --sub_folder="None" \
            --output_dir="semantic_drift/cs101_ti_furniture_chair1_adjust_embedding_v3/rho${rho}_alpha${alpha}/1000"
    done
done
for alpha in 0.5; do
    for rho in 0.2; do
        CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
            --images_folder="evaluation_massive/cs101_ti_furniture_chair1_adjust_embedding_v3/rho${rho}_alpha0.5/1000/<chair>/gen_prompt_cs101_chair" \
            --prompt_file="prompts/gen_prompt_cs101_chair.csv" \
            --num_images=50 \
            --anchor_image_path="cs101/furniture_chair1/0.jpeg" \
            --info="t01" \
            --sub_folder="None" \
            --output_dir="semantic_drift/cs101_ti_furniture_chair1_adjust_embedding_v3/rho${rho}_alpha${alpha}/1000"
    done
done
