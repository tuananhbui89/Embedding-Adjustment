# for RHO in 0.2 0.3 0.8 0.9; do
#     CUDA_VISIBLE_DEVICES=1 python generate_tea.py \
#         --model_path="ckpt/dog_cls" \
#         --concept_name="<new1>" \
#         --output_dir="evaluation_masssive/dog_cls_tea_rho_${RHO}_alpha_0.5" \
#         --guidance_scale=7.0 \
#         --num_images=50 \
#         --prompt_file="prompts/gen_prompt_cs101_dog.csv" \
#         --target_name="a" \
#         --rho=$RHO \
#         --alpha=0.5
# done

# for RHO in 0.2; do
#     CUDA_VISIBLE_DEVICES=1 python generate_tea.py \
#             --model_path="ckpt/teddy_cls" \
#             --concept_name="<new1>" \
#             --output_dir="evaluation_masssive/teddy_cls_tea_rho_${RHO}_alpha_0.5" \
#             --guidance_scale=7.0 \
#             --num_images=50 \
#             --prompt_file="prompts/gen_prompt_cs101_teddybear.csv" \
#             --target_name="a" \
#             --rho=$RHO \
#             --alpha=0.5
# done

# for RHO in 0.2; do
#     CUDA_VISIBLE_DEVICES=1 python generate_tea.py \
#         --model_path="ckpt/barn_cls" \
#         --concept_name="<new1>" \
#         --output_dir="evaluation_masssive/barn_cls_tea_rho_${RHO}_alpha_0.5" \
#         --guidance_scale=7.0 \
#         --num_images=50 \
#         --prompt_file="prompts/gen_prompt_cs101_barn.csv" \
#         --target_name="a" \
#         --rho=$RHO \
#         --alpha=0.5
# done

for RHO in 0.2; do
    CUDA_VISIBLE_DEVICES=1 python generate_tea.py \
        --model_path="ckpt/pot_cls" \
        --concept_name="<new1>" \
        --output_dir="evaluation_masssive/pot_cls_tea_rho_${RHO}_alpha_0.5" \
        --guidance_scale=7.0 \
        --num_images=50 \
        --prompt_file="prompts/gen_prompt_cs101_woodenpot.csv" \
        --target_name="a" \
        --rho=$RHO \
        --alpha=0.5
done

for RHO in 0.2; do
    CUDA_VISIBLE_DEVICES=1 python generate_tea.py \
        --model_path="ckpt/flower_cls" \
        --concept_name="<new1>" \
        --output_dir="evaluation_masssive/flower_cls_tea_rho_${RHO}_alpha_0.5" \
        --guidance_scale=7.0 \
        --num_images=50 \
        --prompt_file="prompts/gen_prompt_cs101_flower.csv" \
        --target_name="a" \
        --rho=$RHO \
        --alpha=0.5
done

for RHO in 0.2; do
    CUDA_VISIBLE_DEVICES=1 python generate_tea.py \
        --model_path="ckpt/chair_cls" \
        --concept_name="<new1>" \
        --output_dir="evaluation_masssive/chair_cls_tea_rho_${RHO}_alpha_0.5" \
        --guidance_scale=7.0 \
        --num_images=50 \
        --prompt_file="prompts/gen_prompt_cs101_chair.csv" \
        --target_name="a" \
        --rho=$RHO \
        --alpha=0.5
done

for RHO in 0.2; do      
    CUDA_VISIBLE_DEVICES=1 python generate_tea.py \
        --model_path="ckpt/table_cls" \
        --concept_name="<new1>" \
        --output_dir="evaluation_masssive/table_cls_tea_rho_${RHO}_alpha_0.5" \
        --guidance_scale=7.0 \
        --num_images=50 \
        --prompt_file="prompts/gen_prompt_cs101_table.csv" \
        --target_name="a" \
        --rho=$RHO \
        --alpha=0.5
done

for RHO in 0.2; do
    CUDA_VISIBLE_DEVICES=1 python generate_tea.py \
        --model_path="ckpt/cat_cls" \
        --concept_name="<new1>" \
        --output_dir="evaluation_masssive/cat_cls_tea_rho_${RHO}_alpha_0.5" \
        --guidance_scale=7.0 \
        --num_images=50 \
        --prompt_file="prompts/gen_prompt_cs101_cat.csv" \
        --target_name="a" \
        --rho=$RHO \
        --alpha=0.5
done

for RHO in 0.2; do
    CUDA_VISIBLE_DEVICES=1 python generate_tea.py \
        --model_path="ckpt/tortoise_cls" \
        --concept_name="<new1>" \
        --output_dir="evaluation_masssive/tortoise_cls_tea_rho_${RHO}_alpha_0.5" \
        --guidance_scale=7.0 \
        --num_images=50 \
        --prompt_file="prompts/gen_prompt_cs101_tortoise.csv" \
        --target_name="a" \
        --rho=$RHO \
        --alpha=0.5
done