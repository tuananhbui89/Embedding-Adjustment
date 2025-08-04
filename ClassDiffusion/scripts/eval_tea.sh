# for RHO in 0.2; do
#     for custom_prompt in objectA objectB full_prompt; do
#         CUDA_VISIBLE_DEVICES=1 python investigate_clip_sim_v2.py \
#             --images_folder="evaluation_masssive/dog_cls_tea_rho_${RHO}_alpha_0.5/" \
#             --prompt_file="prompts/gen_prompt_cs101_dog.csv" \
#             --num_images=50 \
#             --use_custom_prompt=$custom_prompt \
#             --info="use_custom_prompt_$custom_prompt" \
#             --sub_folder="None" \
#             --output_dir="semantic_drift/cs101_class_diffusion_pet_dog1_tea_rho_${RHO}_alpha_0.5/"
#     done
# done

# for RHO in 0.2; do
#     CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
#         --images_folder="evaluation_masssive/dog_cls_tea_rho_${RHO}_alpha_0.5/" \
#         --prompt_file="prompts/gen_prompt_cs101_dog.csv" \
#         --num_images=50 \
#         --anchor_image_path="cs101/pet_dog1/6.jpeg" \
#         --info="t01" \
#         --sub_folder="None" \
#         --output_dir="semantic_drift/cs101_class_diffusion_pet_dog1_tea_rho_${RHO}_alpha_0.5/"

#     CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
#         --images_folder="evaluation_masssive/dog_cls_tea_rho_${RHO}_alpha_0.5/" \
#         --prompt_file="prompts/gen_prompt_cs101_dog.csv" \
#         --num_images=50 \
#         --anchor_image_path="cs101/pet_dog1/6.jpeg" \
#         --info="t01" \
#         --sub_folder="None" \
#         --output_dir="semantic_drift/cs101_class_diffusion_pet_dog1_tea_rho_${RHO}_alpha_0.5/"
# done

# pot - decoritems_woodenpot
for RHO in 0.2; do
    for custom_prompt in objectA objectB full_prompt; do
        CUDA_VISIBLE_DEVICES=1 python investigate_clip_sim_v2.py \
            --images_folder="evaluation_masssive/pot_cls_tea_rho_${RHO}_alpha_0.5/" \
            --prompt_file="prompts/gen_prompt_cs101_woodenpot.csv" \
            --num_images=50 \
            --use_custom_prompt=$custom_prompt \
            --info="use_custom_prompt_$custom_prompt" \
            --sub_folder="None" \
            --output_dir="semantic_drift/cs101_class_diffusion_pot_tea_rho_${RHO}_alpha_0.5/"
    done
done

for RHO in 0.2; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
        --images_folder="evaluation_masssive/pot_cls_tea_rho_${RHO}_alpha_0.5/" \
        --prompt_file="prompts/gen_prompt_cs101_woodenpot.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/decoritems_woodenpot/0.png" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_pot_tea_rho_${RHO}_alpha_0.5/"

    CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
        --images_folder="evaluation_masssive/pot_cls_tea_rho_${RHO}_alpha_0.5/" \
        --prompt_file="prompts/gen_prompt_cs101_woodenpot.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/decoritems_woodenpot/0.png" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_pot_tea_rho_${RHO}_alpha_0.5/"
done

# flower - flower_1
for RHO in 0.2; do
    for custom_prompt in objectA objectB full_prompt; do
        CUDA_VISIBLE_DEVICES=1 python investigate_clip_sim_v2.py \
            --images_folder="evaluation_masssive/flower_cls_tea_rho_${RHO}_alpha_0.5/" \
            --prompt_file="prompts/gen_prompt_cs101_flower.csv" \
            --num_images=50 \
            --use_custom_prompt=$custom_prompt \
            --info="use_custom_prompt_$custom_prompt" \
            --sub_folder="None" \
            --output_dir="semantic_drift/cs101_class_diffusion_flower_tea_rho_${RHO}_alpha_0.5/"
    done
done

for RHO in 0.2; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
        --images_folder="evaluation_masssive/flower_cls_tea_rho_${RHO}_alpha_0.5/" \
        --prompt_file="prompts/gen_prompt_cs101_flower.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/flower_1/0.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_flower_tea_rho_${RHO}_alpha_0.5/"

    CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
        --images_folder="evaluation_masssive/flower_cls_tea_rho_${RHO}_alpha_0.5/" \
        --prompt_file="prompts/gen_prompt_cs101_flower.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/flower_1/0.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_flower_tea_rho_${RHO}_alpha_0.5/"
done

# chair - furniture_chair1  
for RHO in 0.2; do
    for custom_prompt in objectA objectB full_prompt; do
        CUDA_VISIBLE_DEVICES=1 python investigate_clip_sim_v2.py \
            --images_folder="evaluation_masssive/chair_cls_tea_rho_${RHO}_alpha_0.5/" \
            --prompt_file="prompts/gen_prompt_cs101_chair.csv" \
            --num_images=50 \
            --use_custom_prompt=$custom_prompt \
            --info="use_custom_prompt_$custom_prompt" \
            --sub_folder="None" \
            --output_dir="semantic_drift/cs101_class_diffusion_chair_tea_rho_${RHO}_alpha_0.5/"
    done
done

for RHO in 0.2; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
        --images_folder="evaluation_masssive/chair_cls_tea_rho_${RHO}_alpha_0.5/" \
        --prompt_file="prompts/gen_prompt_cs101_chair.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/furniture_chair1/0.jpeg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_chair_tea_rho_${RHO}_alpha_0.5/"

    CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
        --images_folder="evaluation_masssive/chair_cls_tea_rho_${RHO}_alpha_0.5/" \
        --prompt_file="prompts/gen_prompt_cs101_chair.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/furniture_chair1/0.jpeg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_chair_tea_rho_${RHO}_alpha_0.5/"
done

# table - furniture_table1
for RHO in 0.2; do
    for custom_prompt in objectA objectB full_prompt; do
        CUDA_VISIBLE_DEVICES=1 python investigate_clip_sim_v2.py \
            --images_folder="evaluation_masssive/table_cls_tea_rho_${RHO}_alpha_0.5/" \
            --prompt_file="prompts/gen_prompt_cs101_table.csv" \
            --num_images=50 \
            --use_custom_prompt=$custom_prompt \
            --info="use_custom_prompt_$custom_prompt" \
            --sub_folder="None" \
            --output_dir="semantic_drift/cs101_class_diffusion_table_tea_rho_${RHO}_alpha_0.5/"
    done
done

for RHO in 0.2; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
        --images_folder="evaluation_masssive/table_cls_tea_rho_${RHO}_alpha_0.5/" \
        --prompt_file="prompts/gen_prompt_cs101_table.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/furniture_table1/0.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_table_tea_rho_${RHO}_alpha_0.5/"

    CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
        --images_folder="evaluation_masssive/table_cls_tea_rho_${RHO}_alpha_0.5/" \
        --prompt_file="prompts/gen_prompt_cs101_table.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/furniture_table1/0.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_table_tea_rho_${RHO}_alpha_0.5/"
done

# cat - pet_cat1
for RHO in 0.2; do
    for custom_prompt in objectA objectB full_prompt; do
        CUDA_VISIBLE_DEVICES=1 python investigate_clip_sim_v2.py \
            --images_folder="evaluation_masssive/cat_cls_tea_rho_${RHO}_alpha_0.5/" \
            --prompt_file="prompts/gen_prompt_cs101_cat.csv" \
            --num_images=50 \
            --use_custom_prompt=$custom_prompt \
            --info="use_custom_prompt_$custom_prompt" \
            --sub_folder="None" \
            --output_dir="semantic_drift/cs101_class_diffusion_cat_tea_rho_${RHO}_alpha_0.5/"
    done
done

for RHO in 0.2; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
        --images_folder="evaluation_masssive/cat_cls_tea_rho_${RHO}_alpha_0.5/" \
        --prompt_file="prompts/gen_prompt_cs101_cat.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/pet_cat1/jeanie-de-klerk-bhonzdJMVjY-unsplash.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_cat_tea_rho_${RHO}_alpha_0.5/"

    CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
        --images_folder="evaluation_masssive/cat_cls_tea_rho_${RHO}_alpha_0.5/" \
        --prompt_file="prompts/gen_prompt_cs101_cat.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/pet_cat1/jeanie-de-klerk-bhonzdJMVjY-unsplash.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_cat_tea_rho_${RHO}_alpha_0.5/"
done

# teddy - plushie_teddybear
for RHO in 0.2; do
    for custom_prompt in objectA objectB full_prompt; do
        CUDA_VISIBLE_DEVICES=1 python investigate_clip_sim_v2.py \
            --images_folder="evaluation_masssive/teddy_cls_tea_rho_${RHO}_alpha_0.5/" \
            --prompt_file="prompts/gen_prompt_cs101_teddybear.csv" \
            --num_images=50 \
            --use_custom_prompt=$custom_prompt \
            --info="use_custom_prompt_$custom_prompt" \
            --sub_folder="None" \
            --output_dir="semantic_drift/cs101_class_diffusion_teddy_tea_rho_${RHO}_alpha_0.5/"
    done
done

for RHO in 0.2; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
        --images_folder="evaluation_masssive/teddy_cls_tea_rho_${RHO}_alpha_0.5/" \
        --prompt_file="prompts/gen_prompt_cs101_teddybear.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/plushie_teddybear/marina-shatskih-6MDi8o6VYHg-unsplash.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_teddy_tea_rho_${RHO}_alpha_0.5/"

    CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
        --images_folder="evaluation_masssive/teddy_cls_tea_rho_${RHO}_alpha_0.5/" \
        --prompt_file="prompts/gen_prompt_cs101_teddybear.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/plushie_teddybear/marina-shatskih-6MDi8o6VYHg-unsplash.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_teddy_tea_rho_${RHO}_alpha_0.5/"
done

# tortoise - plushie_tortoise
for RHO in 0.2; do
    for custom_prompt in objectA objectB full_prompt; do
        CUDA_VISIBLE_DEVICES=1 python investigate_clip_sim_v2.py \
            --images_folder="evaluation_masssive/tortoise_cls_tea_rho_${RHO}_alpha_0.5/" \
            --prompt_file="prompts/gen_prompt_cs101_tortoise.csv" \
            --num_images=50 \
            --use_custom_prompt=$custom_prompt \
            --info="use_custom_prompt_$custom_prompt" \
            --sub_folder="None" \
            --output_dir="semantic_drift/cs101_class_diffusion_tortoise_tea_rho_${RHO}_alpha_0.5/"
    done
done

for RHO in 0.2; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
        --images_folder="evaluation_masssive/tortoise_cls_tea_rho_${RHO}_alpha_0.5/" \
        --prompt_file="prompts/gen_prompt_cs101_tortoise.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/plushie_tortoise/4.png" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_tortoise_tea_rho_${RHO}_alpha_0.5/"

    CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
        --images_folder="evaluation_masssive/tortoise_cls_tea_rho_${RHO}_alpha_0.5/" \
        --prompt_file="prompts/gen_prompt_cs101_tortoise.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/plushie_tortoise/4.png" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_tortoise_tea_rho_${RHO}_alpha_0.5/"
done

# barn - scene_barn
for RHO in 0.2; do
    for custom_prompt in objectA objectB full_prompt; do
        CUDA_VISIBLE_DEVICES=1 python investigate_clip_sim_v2.py \
            --images_folder="evaluation_masssive/barn_cls_tea_rho_${RHO}_alpha_0.5/" \
            --prompt_file="prompts/gen_prompt_cs101_barn.csv" \
            --num_images=50 \
            --use_custom_prompt=$custom_prompt \
            --info="use_custom_prompt_$custom_prompt" \
            --sub_folder="None" \
            --output_dir="semantic_drift/cs101_class_diffusion_barn_tea_rho_${RHO}_alpha_0.5/"
    done
done

for RHO in 0.2; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
        --images_folder="evaluation_masssive/barn_cls_tea_rho_${RHO}_alpha_0.5/" \
        --prompt_file="prompts/gen_prompt_cs101_barn.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/scene_barn/2.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_barn_tea_rho_${RHO}_alpha_0.5/"

    CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
        --images_folder="evaluation_masssive/barn_cls_tea_rho_${RHO}_alpha_0.5/" \
        --prompt_file="prompts/gen_prompt_cs101_barn.csv" \
        --num_images=50 \
        --anchor_image_path="cs101/scene_barn/2.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/cs101_class_diffusion_barn_tea_rho_${RHO}_alpha_0.5/"
done
