export MODEL_DIR="black-forest-labs/FLUX.1-dev" # your flux path
export OUTPUT_DIR="./models/subject_model_with_text_encoder"  # your save path
export CONFIG="./single_gpu_config.yaml"  # Use RTX 4000 optimized config
export TRAIN_DATA="./examples/subject.jsonl"  # your data jsonl file
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
    --num_train_epochs=1000 \
    --validation_steps=50 \
    --checkpointing_steps=50 \
    --spatial_test_images None \
    --subject_test_images "./examples/subject_data/3.png" \
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