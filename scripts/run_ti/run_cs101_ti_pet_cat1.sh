#!/bin/bash

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 textual_inversion.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --train_data_dir="cs101/pet_cat1" \
  --learnable_property="object" \
  --placeholder_token="<cat>" --initializer_token="cat" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="outputs/cs101_ti_pet_cat1" \
  --validation_steps=100 \
  --save_steps=100 \
  --checkpointing_steps=10000 \
  --mixed_precision="no" \
  --validation_prompt="a photo of a <cat> wearing glasses and writing on a red notebook"

CUDA_VISIBLE_DEVICES=0 python generate_textual_inversion.py \
    --method="distill_ti" \
    --model_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --concept_name="<cat>" \
    --output_dir="evaluation_massive/cs101_ti_pet_cat1_steps_1000" \
    --guidance_scale=7.0 \
    --embedding_path="outputs/cs101_ti_pet_cat1/learned_embeds-steps-1000.bin" \
    --max_train_steps=1000 \
    --start_steps=0 \
    --num_images=50 \
    --prompt_file="prompts/gen_prompt_cs101_cat.csv"

for custom_prompt in 'objectA' 'objectB' 'full_prompt'; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
        --images_folder="evaluation_massive/cs101_ti_pet_cat1_steps_1000/<cat>/gen_prompt_cs101_cat" \
        --prompt_file="prompts/gen_prompt_cs101_cat.csv" \
        --num_images=50 \
        --use_custom_prompt=$custom_prompt \
        --info="use_custom_prompt_${custom_prompt}" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_ti_pet_cat1_steps_1000/1000"
done

CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
    --images_folder="evaluation_massive/cs101_ti_pet_cat1_steps_1000/<cat>/gen_prompt_cs101_cat" \
    --prompt_file="prompts/gen_prompt_cs101_cat.csv" \
    --num_images=50 \
    --anchor_image_path="cs101/pet_cat1/jeanie-de-klerk-bhonzdJMVjY-unsplash.jpg" \
    --info="t01" \
    --sub_folder="None" \
    --output_dir="semantic_drift/cs101_ti_pet_cat1_steps_1000/1000"


CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
    --images_folder="evaluation_massive/cs101_ti_pet_cat1_steps_1000/<cat>/gen_prompt_cs101_cat" \
    --prompt_file="prompts/gen_prompt_cs101_cat.csv" \
    --num_images=50 \
    --anchor_image_path="cs101/pet_cat1/jeanie-de-klerk-bhonzdJMVjY-unsplash.jpg" \
    --info="t01" \
    --sub_folder="None" \
    --output_dir="semantic_drift/cs101_ti_pet_cat1_steps_1000/1000"


