for setting in clock tshirt penguin rc_car oranges; do
    for custom_prompt in full_prompt; do
        CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
            --images_folder="evaluation_massive/omini_subject/$setting" \
            --prompt_file="prompts/eval_prompt_omini_$setting.csv" \
            --num_images=50 \
            --use_custom_prompt=$custom_prompt \
            --info="use_custom_prompt_$custom_prompt" \
            --sub_folder="None" \
            --output_dir="semantic_drift/omini_subject/$setting"
    done

    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
        --images_folder="evaluation_massive/omini_subject/$setting" \
        --prompt_file="prompts/eval_prompt_omini_$setting.csv" \
        --num_images=50 \
        --anchor_image_path="assets/$setting.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/omini_subject/$setting"

    CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
        --images_folder="evaluation_massive/omini_subject/$setting" \
        --prompt_file="prompts/eval_prompt_omini_$setting.csv" \
        --num_images=50 \
        --anchor_image_path="assets/$setting.jpg" \
        --info="t01" \
        --sub_folder="None" \
        --output_dir="semantic_drift/omini_subject/$setting"
done

# version 1
for setting in clock tshirt penguin rc_car oranges; do
    for param in 0.3_0.5; do
        for custom_prompt in full_prompt; do
            CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
                --images_folder="evaluation_massive/omini_subject_tea/${setting}_${param}" \
                --prompt_file="prompts/eval_prompt_omini_$setting.csv" \
                --num_images=50 \
                --use_custom_prompt=$custom_prompt \
                --info="use_custom_prompt_$custom_prompt" \
                --sub_folder="None" \
                --output_dir="semantic_drift/omini_subject_tea/${setting}_${param}"
        done

        CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_image.py \
            --images_folder="evaluation_massive/omini_subject_tea/${setting}_${param}" \
            --prompt_file="prompts/eval_prompt_omini_$setting.csv" \
            --num_images=50 \
            --anchor_image_path="assets/$setting.jpg" \
            --info="t01" \
            --sub_folder="None" \
            --output_dir="semantic_drift/omini_subject_tea/${setting}_${param}"

        CUDA_VISIBLE_DEVICES=0 python investigate_dino_sim_image.py \
            --images_folder="evaluation_massive/omini_subject_tea/${setting}_${param}" \
            --prompt_file="prompts/eval_prompt_omini_$setting.csv" \
            --num_images=50 \
            --anchor_image_path="assets/$setting.jpg" \
            --info="t01" \
            --sub_folder="None" \
                --output_dir="semantic_drift/omini_subject_tea/${setting}_${param}"
    done
done
