#!/bin/bash
# evaluation_massive/celebA_342_db_lora_lr1e-4_rank4_train_text_encoder/1800/sks man/gen_prompt_actions/0_0.png
# --anchor_image_path="celebA/342/416.jpg" \
# --info="416_step_$STEP" \

for custom_prompt in objectA objectB full_prompt; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
        --images_folder="evaluation_massive/celebA_342_db_lora_lr1e-4_rank4_train_text_encoder/1800/sks man/gen_prompt_actions" \
        --prompt_file="prompts/gen_prompt_actions.csv" \
        --num_images=100 \
        --use_custom_prompt=$custom_prompt \
        --info="use_custom_prompt_$custom_prompt" \
        --sub_folder="None" \
        --output_dir="semantic_drift/celebA_342_db_lora_lr1e-4_rank4_train_text_encoder/1800"
done
CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
    --images_folder="evaluation_massive/celebA_342_db_lora_lr1e-4_rank4_train_text_encoder/1800/sks man/gen_prompt_actions" \
    --prompt_file="prompts/gen_prompt_actions.csv" \
    --num_images=100 \
    --anchor_image_path="celebA/342/416.jpg" \
    --info="416_step_1800" \
    --sub_folder="None" \
    --output_dir="semantic_drift/celebA_342_db_lora_lr1e-4_rank4_train_text_encoder/1800"
