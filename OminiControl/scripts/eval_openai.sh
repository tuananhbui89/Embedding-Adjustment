eval_model="gpt-4o-mini"
# eval_model="gpt-5"


for setting in clock; do

    CUDA_VISIBLE_DEVICES=0 python eval_openai.py \
        --images_folder="evaluation_massive/omini_subject/$setting" \
        --prompt_file="prompts/eval_prompt_omini_$setting.csv" \
        --num_images=50 \
        --anchor_image_path="assets/$setting.jpg" \
        --output_dir="semantic_drift/omini_subject/$setting" \
        --eval_model=$eval_model \
        --info="system_prompt_v4" \
        --system_prompt_file="eval_system_prompt_v4.txt"
done

for setting in clock; do
    for param in 0.3_0.5; do

        CUDA_VISIBLE_DEVICES=0 python eval_openai.py \
            --images_folder="evaluation_massive/omini_subject_tea/${setting}_${param}" \
            --prompt_file="prompts/eval_prompt_omini_$setting.csv" \
            --num_images=50 \
            --anchor_image_path="assets/$setting.jpg" \
            --output_dir="semantic_drift/omini_subject_tea/${setting}_${param}" \
            --eval_model=$eval_model \
            --info="system_prompt_v4" \
            --system_prompt_file="eval_system_prompt_v4.txt"
    done
done