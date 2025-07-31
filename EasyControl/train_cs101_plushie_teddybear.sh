export MODEL_DIR="black-forest-labs/FLUX.1-dev" # your flux path
export OUTPUT_DIR="./models/cs101_plushie_teddybear"  # your save path
export CONFIG="./single_gpu_config.yaml"  # Use RTX 4000 optimized config
export TRAIN_DATA="./cs101/plushie_teddybear.jsonl"  # your data jsonl file
export LOG_PATH="$OUTPUT_DIR/log"

# Memory optimization: Limit GPU memory growth
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# RTX 4000 series specific NCCL settings
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file $CONFIG train.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --cond_size=384 \
    --noise_size=768 \
    --subject_column="source" \
    --spatial_column="None" \
    --target_column="target" \
    --caption_column="caption" \
    --ranks 64 \
    --network_alphas 64 \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --mixed_precision="bf16" \
    --train_data_dir=$TRAIN_DATA \
    --learning_rate=1e-4 \
    --train_batch_size=1 \
    --validation_prompt "An SKS in the city." \
    --num_train_epochs=300 \
    --validation_steps=50 \
    --checkpointing_steps=50 \
    --spatial_test_images None \
    --subject_test_images "cs101/plushie_teddybear/marina-shatskih-6MDi8o6VYHg-unsplash.jpg" \
    --test_h 768 \
    --test_w 768 \
    --num_validation_images=1 \
    --train_text_encoder_one_only \
    --text_encoder_lr=5e-6 \
    --text_encoder_rank=32 \
    --text_encoder_alpha=32 \
    --gradient_checkpointing \
    --dataloader_num_workers=2 \
    --max_grad_norm=1.0 

python infer_subject_with_text_encoder.py \
    --checkpoint_step=checkpoint-1000 \
    --trained_model_dir="./models/cs101_plushie_teddybear" \
    --subject_image_path="cs101/plushie_teddybear/marina-shatskih-6MDi8o6VYHg-unsplash.jpg" \
    --subject_token="SKS" \
    --prompt_file_path="prompts/gen_prompt_cs101.csv" \
    --height=768 \
    --width=768 \
    --cond_size=384 \
    --num_images=50 \
    --output_path="evaluation_massive/output_cs101_plushie_teddybear"


for rho in 0.3; do
    for alpha in 0.5; do
        python infer_subject_with_tea.py \
            --checkpoint_step=checkpoint-1000 \
            --trained_model_dir=$OUTPUT_DIR \
            --subject_image_path="cs101/plushie_teddybear/marina-shatskih-6MDi8o6VYHg-unsplash.jpg" \
            --subject_token="SKS" \
            --prompt_file_path="prompts/gen_prompt_cs101.csv" \
            --height=768 \
            --width=768 \
            --cond_size=384 \
            --rho=$rho \
            --alpha=$alpha \
            --target_token="teddy bear" \
            --num_images=50 \
            --output_path="evaluation_massive/output_cs101_plushie_teddybear_tea_teddybear_${rho}_${alpha}"
    done
done


for RHO in 0.3; do
  for custom_prompt in objectA objectB full_prompt; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
        --images_folder="evaluation_massive/output_cs101_plushie_teddybear_tea_${RHO}_0.5" \
        --prompt_file="prompts/gen_prompt_cs101_teddybear.csv" \
        --num_images=50 \
        --use_custom_prompt=$custom_prompt \
        --info="use_custom_prompt_$custom_prompt" \
        --sub_folder="None" \
        --output_dir="semantic_drift/output_cs101_plushie_teddybear_tea_${RHO}_0.5"
  done
done

for RHO in 0.3; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
        --images_folder="evaluation_massive/output_cs101_plushie_teddybear_tea_${RHO}_0.5" \
        --prompt_file="prompts/gen_prompt_cs101_teddybear.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/plushie_teddybear/marina-shatskih-6MDi8o6VYHg-unsplash.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/output_cs101_plushie_teddybear_tea_${RHO}_0.5"
done

for RHO in 0.3; do
    CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
        --images_folder="evaluation_massive/output_cs101_plushie_teddybear_tea_${RHO}_0.5" \
        --prompt_file="prompts/gen_prompt_cs101_teddybear.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/plushie_teddybear/marina-shatskih-6MDi8o6VYHg-unsplash.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/output_cs101_plushie_teddybear_tea_${RHO}_0.5"
done

baseline teddybear

for custom_prompt in objectA objectB full_prompt; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
        --images_folder="evaluation_massive/output_cs101_plushie_teddybear" \
        --prompt_file="prompts/gen_prompt_cs101_teddybear.csv" \
        --num_images=50 \
        --use_custom_prompt=$custom_prompt \
        --info="use_custom_prompt_$custom_prompt" \
        --sub_folder="None" \
        --output_dir="semantic_drift/output_cs101_plushie_teddybear"
done


CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
    --images_folder="evaluation_massive/output_cs101_plushie_teddybear" \
    --prompt_file="prompts/gen_prompt_cs101_teddybear.csv" \
    --num_images=50 \
    --anchor_image_path="cs101/plushie_teddybear/marina-shatskih-6MDi8o6VYHg-unsplash.jpg" \
    --info="t01" \
    --sub_folder="None" \
    --output_dir="semantic_drift/output_cs101_plushie_teddybear"


CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
    --images_folder="evaluation_massive/output_cs101_plushie_teddybear" \
    --prompt_file="prompts/gen_prompt_cs101_teddybear.csv" \
    --num_images=50 \
    --anchor_image_path="cs101/plushie_teddybear/marina-shatskih-6MDi8o6VYHg-unsplash.jpg" \
    --info="t01" \
    --sub_folder="None" \
    --output_dir="semantic_drift/output_cs101_plushie_teddybear"
