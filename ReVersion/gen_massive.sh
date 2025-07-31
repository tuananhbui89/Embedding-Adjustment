your_template_name='painted_on_examples'
CUDA_VISIBLE_DEVICES=1 python inference.py \
--model_id ./experiments/painted_on \
--template_name $your_template_name \
--placeholder_string "<R>" \
--num_samples 50 \
--guidance_scale 7.5 \
--only_load_embeds \
--output_dir "evaluation_massive/painted_on" \
--prompt_file "prompts_painted_on.csv"

your_template_name='painted_on_examples'
for RHO in 0.2 0.3; do
    CUDA_VISIBLE_DEVICES=1 python inference_tea.py \
    --model_id ./experiments/painted_on \
    --template_name $your_template_name \
    --placeholder_string "<R>" \
    --num_samples 50 \
    --guidance_scale 7.5 \
    --only_load_embeds \
    --output_dir "evaluation_massive/painted_on_tea_${RHO}_0.5" \
    --rho $RHO \
    --alpha 0.5 \
    --target_word "painted on" \
    --prompt_file "prompts_painted_on.csv"
done

your_template_name='inside_examples'
CUDA_VISIBLE_DEVICES=1 python inference.py \
--model_id ./experiments/inside \
--template_name $your_template_name \
--placeholder_string "<R>" \
--num_samples 50 \
--guidance_scale 7.5 \
--only_load_embeds \
--output_dir "evaluation_massive/inside" \
--prompt_file "prompts_inside.csv"

your_template_name='inside_examples'
for RHO in 0.2 0.3; do
    CUDA_VISIBLE_DEVICES=1 python inference_tea.py \
    --model_id ./experiments/inside \
    --template_name $your_template_name \
    --placeholder_string "<R>" \
    --num_samples 50 \
    --guidance_scale 7.5 \
    --only_load_embeds \
    --output_dir "evaluation_massive/inside_tea_${RHO}_0.5" \
    --rho $RHO \
    --alpha 0.5 \
    --target_word "inside" \
    --prompt_file "prompts_inside.csv"
done

your_template_name='carved_by_examples'
CUDA_VISIBLE_DEVICES=1 python inference.py \
--model_id ./experiments/carved_by \
--template_name $your_template_name \
--placeholder_string "<R>" \
--num_samples 50 \
--guidance_scale 7.5 \
--only_load_embeds \
--output_dir "evaluation_massive/carved_by" \
--prompt_file "prompts_carved_by.csv"

your_template_name='carved_by_examples'
for RHO in 0.2 0.3; do
    CUDA_VISIBLE_DEVICES=1 python inference_tea.py \
    --model_id ./experiments/carved_by \
    --template_name $your_template_name \
    --placeholder_string "<R>" \
    --num_samples 50 \
    --guidance_scale 7.5 \
    --only_load_embeds \
    --output_dir "evaluation_massive/carved_by_tea_v2_${RHO}_0.5" \
    --rho $RHO \
    --alpha 0.5 \
    --target_word "carved by" \
    --prompt_file "prompts_carved_by.csv"
done


for custom_prompt in objectA objectB full_prompt; do
CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
    --images_folder="evaluation_massive/painted_on" \
    --prompt_file="prompts_painted_on.csv" \
    --num_images=50 \
    --use_custom_prompt=$custom_prompt \
    --info="use_custom_prompt_$custom_prompt" \
    --sub_folder="None" \
    --output_dir="semantic_drift/output_painted_on"
done

for RHO in 0.2 0.3; do
  for custom_prompt in objectA objectB full_prompt; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
        --images_folder="evaluation_massive/painted_on_tea_${RHO}_0.5" \
        --prompt_file="prompts_painted_on.csv" \
        --num_images=50 \
        --use_custom_prompt=$custom_prompt \
        --info="use_custom_prompt_$custom_prompt" \
        --sub_folder="None" \
        --output_dir="semantic_drift/output_painted_on_tea_${RHO}_0.5"
  done
done

for custom_prompt in objectA objectB full_prompt; do
CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
    --images_folder="evaluation_massive/inside" \
    --prompt_file="prompts_inside.csv" \
    --num_images=50 \
    --use_custom_prompt=$custom_prompt \
    --info="use_custom_prompt_$custom_prompt" \
    --sub_folder="None" \
    --output_dir="semantic_drift/output_inside"
done

for RHO in 0.2 0.3; do
  for custom_prompt in objectA objectB full_prompt; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
        --images_folder="evaluation_massive/inside_tea_${RHO}_0.5" \
        --prompt_file="prompts_inside.csv" \
        --num_images=50 \
        --use_custom_prompt=$custom_prompt \
        --info="use_custom_prompt_$custom_prompt" \
        --sub_folder="None" \
        --output_dir="semantic_drift/output_inside_tea_${RHO}_0.5"
  done
done

for custom_prompt in objectA objectB full_prompt; do
CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
    --images_folder="evaluation_massive/carved_by" \
    --prompt_file="prompts_carved_by.csv" \
    --num_images=50 \
    --use_custom_prompt=$custom_prompt \
    --info="use_custom_prompt_$custom_prompt" \
    --sub_folder="None" \
    --output_dir="semantic_drift/output_carved_by"
done

for RHO in 0.2 0.3; do
  for custom_prompt in objectA objectB full_prompt; do
    CUDA_VISIBLE_DEVICES=0 python investigate_clip_sim_v2.py \
        --images_folder="evaluation_massive/carved_by_tea_v2_${RHO}_0.5" \
        --prompt_file="prompts_carved_by.csv" \
        --num_images=50 \
        --use_custom_prompt=$custom_prompt \
        --info="use_custom_prompt_$custom_prompt" \
        --sub_folder="None" \
        --output_dir="semantic_drift/output_carved_by_tea_v2_${RHO}_0.5"
  done
done