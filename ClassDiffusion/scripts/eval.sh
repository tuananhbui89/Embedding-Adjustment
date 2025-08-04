
# for custom_prompt in objectA objectB full_prompt; do
#     CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
#         --images_folder="evaluation_masssive/dog_cls/" \
#         --prompt_file="prompts/gen_prompt_cs101_dog.csv" \
#         --num_images=50 \
#         --use_custom_prompt=$custom_prompt \
#         --info="use_custom_prompt_$custom_prompt" \
#         --sub_folder="None" \
#         --output_dir="semantic_drift/cs101_class_diffusion_pet_dog1/"
# done

# CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
#         --images_folder="evaluation_masssive/dog_cls/" \
#         --prompt_file="prompts/gen_prompt_cs101_dog.csv" \
#         --num_images=50 \
#         --anchor_image_path="cs101/pet_dog1/6.jpeg" \
#         --info="t01" \
#         --sub_folder="None" \
#         --output_dir="semantic_drift/cs101_class_diffusion_pet_dog1/"

# CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
#         --images_folder="evaluation_masssive/dog_cls/" \
#         --prompt_file="prompts/gen_prompt_cs101_dog.csv" \
#         --num_images=50 \
#         --anchor_image_path="cs101/pet_dog1/6.jpeg" \
#         --info="t01" \
#         --sub_folder="None" \
#         --output_dir="semantic_drift/cs101_class_diffusion_pet_dog1/"

# pot cs101/decoritems_woodenpot/0.png

for custom_prompt in objectA objectB full_prompt; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
        --images_folder="evaluation_masssive/pot_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_woodenpot.csv" \
        --num_images=50 \
        --use_custom_prompt=$custom_prompt \
        --info="use_custom_prompt_$custom_prompt" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_pot/"
done

CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
        --images_folder="evaluation_masssive/pot_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_woodenpot.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/decoritems_woodenpot/0.png" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_pot/"

CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
        --images_folder="evaluation_masssive/pot_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_woodenpot.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/decoritems_woodenpot/0.png" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_pot/"

# flower anchor_image_path="cs101/flower_1/0.jpg"

for custom_prompt in objectA objectB full_prompt; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
        --images_folder="evaluation_masssive/flower_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_flower.csv" \
        --num_images=50 \
        --use_custom_prompt=$custom_prompt \
        --info="use_custom_prompt_$custom_prompt" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_flower/"
done

CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
        --images_folder="evaluation_masssive/flower_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_flower.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/flower_1/0.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_flower/"

CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
        --images_folder="evaluation_masssive/flower_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_flower.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/flower_1/0.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_flower/"

# chair anchor_image_path="cs101/furniture_chair1/0.jpeg"


for custom_prompt in objectA objectB full_prompt; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
        --images_folder="evaluation_masssive/chair_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_chair.csv" \
        --num_images=50 \
        --use_custom_prompt=$custom_prompt \
        --info="use_custom_prompt_$custom_prompt" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_chair/"
done

CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
        --images_folder="evaluation_masssive/chair_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_chair.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/furniture_chair1/0.jpeg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_chair/"

CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
        --images_folder="evaluation_masssive/chair_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_chair.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/furniture_chair1/0.jpeg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_chair/"

# table anchor_image_path="cs101/furniture_table1/0.jpg"

for custom_prompt in objectA objectB full_prompt; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
        --images_folder="evaluation_masssive/table_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_table.csv" \
        --num_images=50 \
        --use_custom_prompt=$custom_prompt \
        --info="use_custom_prompt_$custom_prompt" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_table/"
done

CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
        --images_folder="evaluation_masssive/table_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_table.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/furniture_table1/0.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_table/"

CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
        --images_folder="evaluation_masssive/table_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_table.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/furniture_table1/0.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_table/"

# cat anchor_image_path="cs101/pet_cat1/jeanie-de-klerk-bhonzdJMVjY-unsplash.jpg"

for custom_prompt in objectA objectB full_prompt; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
        --images_folder="evaluation_masssive/cat_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_cat.csv" \
        --num_images=50 \
        --use_custom_prompt=$custom_prompt \
        --info="use_custom_prompt_$custom_prompt" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_cat/"
done

CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
        --images_folder="evaluation_masssive/cat_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_cat.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/pet_cat1/jeanie-de-klerk-bhonzdJMVjY-unsplash.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_cat/"        

CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
        --images_folder="evaluation_masssive/cat_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_cat.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/pet_cat1/jeanie-de-klerk-bhonzdJMVjY-unsplash.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_cat/"

# teddy anchor_image_path="cs101/plushie_teddybear/marina-shatskih-6MDi8o6VYHg-unsplash.jpg"

for custom_prompt in objectA objectB full_prompt; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
        --images_folder="evaluation_masssive/teddy_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_teddybear.csv" \
        --num_images=50 \
        --use_custom_prompt=$custom_prompt \
        --info="use_custom_prompt_$custom_prompt" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_teddy/"
done

CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
        --images_folder="evaluation_masssive/teddy_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_teddybear.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/plushie_teddybear/marina-shatskih-6MDi8o6VYHg-unsplash.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_teddy/"

CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
        --images_folder="evaluation_masssive/teddy_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_teddybear.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/plushie_teddybear/marina-shatskih-6MDi8o6VYHg-unsplash.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_teddy/"

# tortoise anchor_image_path="cs101/plushie_tortoise/4.png"

for custom_prompt in objectA objectB full_prompt; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
        --images_folder="evaluation_masssive/tortoise_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_tortoise.csv" \
        --num_images=50 \
        --use_custom_prompt=$custom_prompt \
        --info="use_custom_prompt_$custom_prompt" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_tortoise/"
done

CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
        --images_folder="evaluation_masssive/tortoise_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_tortoise.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/plushie_tortoise/4.png" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_tortoise/"

CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
        --images_folder="evaluation_masssive/tortoise_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_tortoise.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/plushie_tortoise/4.png" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_tortoise/"

# barn anchor_image_path="cs101/scene_barn/2.jpg"

for custom_prompt in objectA objectB full_prompt; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
        --images_folder="evaluation_masssive/barn_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_barn.csv" \
        --num_images=50 \
        --use_custom_prompt=$custom_prompt \
        --info="use_custom_prompt_$custom_prompt" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_barn/"
done

CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
        --images_folder="evaluation_masssive/barn_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_barn.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/scene_barn/2.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_barn/"

CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
        --images_folder="evaluation_masssive/barn_cls/" \
        --prompt_file="prompts/gen_prompt_cs101_barn.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/scene_barn/2.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_barn/"
