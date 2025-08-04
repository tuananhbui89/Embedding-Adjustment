# export CLS_TOKEN="dog"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export OUTPUT_DIR="./ckpt/${CLS_TOKEN}_cls"
# export INSTANCE_DIR="./cs101/pet_dog1"

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=29503 train_class_diffusion.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a photo of a <new1> ${CLS_TOKEN}"  \
#   --resolution=512  \
#   --train_batch_size=2  \
#   --learning_rate=1e-5  \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --scale_lr \
#   --hflip  \
#   --modifier_token "<new1>" \
#   --no_safe_serialization \
#   --use_spl \
#   --spl_weight=1 \
#   --cls_token "${CLS_TOKEN}" \
#   --validation_steps=50 \
#   --validation_prompt="a <new1> ${CLS_TOKEN} is swimming" \


# export CLS_TOKEN="teddy"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export OUTPUT_DIR="./ckpt/${CLS_TOKEN}_cls"
# export INSTANCE_DIR="./cs101/plushie_teddybear"

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=29503 train_class_diffusion.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a photo of a <new1> ${CLS_TOKEN}"  \
#   --resolution=512  \
#   --train_batch_size=2  \
#   --learning_rate=1e-5  \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --scale_lr \
#   --hflip  \
#   --modifier_token "<new1>" \
#   --no_safe_serialization \
#   --use_spl \
#   --spl_weight=1 \
#   --cls_token "${CLS_TOKEN}" \
#   --validation_steps=50 \
#   --validation_prompt="a <new1> ${CLS_TOKEN} is swimming" \

# export CLS_TOKEN="barn"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export OUTPUT_DIR="./ckpt/${CLS_TOKEN}_cls"
# export INSTANCE_DIR="./cs101/scene_barn"

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=29503 train_class_diffusion.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a photo of a <new1> ${CLS_TOKEN}"  \
#   --resolution=512  \
#   --train_batch_size=2  \
#   --learning_rate=1e-5  \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --scale_lr \
#   --hflip  \
#   --modifier_token "<new1>" \
#   --no_safe_serialization \
#   --use_spl \
#   --spl_weight=1 \
#   --cls_token "${CLS_TOKEN}" \
#   --validation_steps=50 \
#   --validation_prompt="a <new1> ${CLS_TOKEN} is swimming" \

# export CLS_TOKEN="pot"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export OUTPUT_DIR="./ckpt/${CLS_TOKEN}_cls"
# export INSTANCE_DIR="./cs101/decoritems_woodenpot"

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=29503 train_class_diffusion.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a photo of a <new1> ${CLS_TOKEN}"  \
#   --resolution=512  \
#   --train_batch_size=2  \
#   --learning_rate=1e-5  \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --scale_lr \
#   --hflip  \
#   --modifier_token "<new1>" \
#   --no_safe_serialization \
#   --use_spl \
#   --spl_weight=1 \
#   --cls_token "${CLS_TOKEN}" \
#   --validation_steps=50 \
#   --validation_prompt="a <new1> ${CLS_TOKEN} on a table" \

# export CLS_TOKEN="flower"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export OUTPUT_DIR="./ckpt/${CLS_TOKEN}_cls"
# export INSTANCE_DIR="./cs101/flower_1"

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=29503 train_class_diffusion.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a photo of a <new1> ${CLS_TOKEN}"  \
#   --resolution=512  \
#   --train_batch_size=2  \
#   --learning_rate=1e-5  \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --scale_lr \
#   --hflip  \
#   --modifier_token "<new1>" \
#   --no_safe_serialization \
#   --use_spl \
#   --spl_weight=1 \
#   --cls_token "${CLS_TOKEN}" \
#   --validation_steps=50 \
#   --validation_prompt="a <new1> ${CLS_TOKEN} on a table" \

export CLS_TOKEN="chair"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./ckpt/${CLS_TOKEN}_cls"
export INSTANCE_DIR="./cs101/furniture_chair1"

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=29503 train_class_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of a <new1> ${CLS_TOKEN}"  \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --scale_lr \
  --hflip  \
  --modifier_token "<new1>" \
  --no_safe_serialization \
  --use_spl \
  --spl_weight=1 \
  --cls_token "${CLS_TOKEN}" \
  --validation_steps=50 \
  --validation_prompt="a <new1> ${CLS_TOKEN} in VanGogh style" \

export CLS_TOKEN="table"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./ckpt/${CLS_TOKEN}_cls"
export INSTANCE_DIR="./cs101/furniture_table1"

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=29503 train_class_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of a <new1> ${CLS_TOKEN}"  \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --scale_lr \
  --hflip  \
  --modifier_token "<new1>" \
  --no_safe_serialization \
  --use_spl \
  --spl_weight=1 \
  --cls_token "${CLS_TOKEN}" \
  --validation_steps=50 \
  --validation_prompt="a <new1> ${CLS_TOKEN} in VanGogh style" \

export CLS_TOKEN="cat"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./ckpt/${CLS_TOKEN}_cls"
export INSTANCE_DIR="./cs101/pet_cat1"

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=29503 train_class_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of a <new1> ${CLS_TOKEN}"  \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --scale_lr \
  --hflip  \
  --modifier_token "<new1>" \
  --no_safe_serialization \
  --use_spl \
  --spl_weight=1 \
  --cls_token "${CLS_TOKEN}" \
  --validation_steps=50 \
  --validation_prompt="a <new1> ${CLS_TOKEN} is swimming" \

export CLS_TOKEN="tortoise"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./ckpt/${CLS_TOKEN}_cls"
export INSTANCE_DIR="./cs101/plushie_tortoise"

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=29503 train_class_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of a <new1> ${CLS_TOKEN}"  \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --scale_lr \
  --hflip  \
  --modifier_token "<new1>" \
  --no_safe_serialization \
  --use_spl \
  --spl_weight=1 \
  --cls_token "${CLS_TOKEN}" \
  --validation_steps=50 \
  --validation_prompt="a <new1> ${CLS_TOKEN} is swimming" \