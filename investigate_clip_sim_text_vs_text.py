import pandas as pd
import os
from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision.transforms.functional import pil_to_tensor, resize
import torch

metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")
metric.to('cuda')

os.makedirs('semantic_drift', exist_ok=True)

exp_name = 'celebA_342_ti'
info = 'full_prompt'

output_file = f'semantic_drift/clip_alignment_text_vs_text_{exp_name}_{info}.csv'

prompts_A = [
    "a man",
    "a woman",
    "Henry Cavill",
    "a toy", 
    "a bus", 
    "a banana"
]

prompts_B = [
    "handshaking",
    "handshaking with a man",
    "handshaking with a woman",
    "handshaking with an old man",
    "handshaking with a kid",
    "holding",
    "holding a dog",
    "holding a cat",
    "holding a red book",
    "holding a red phone",
    "sitting",
    "sitting on a red chair",
    "lying",
    "lying on a red bed",
    "writing",
    "writing in a red notebook",
    "drinking",
    "drinking a Coco Cola can",
    "lifting",
    "lifting weights",
    "cycling",
    "kicking",
    "kicking a football",
    "playing",
    "playing a guitar",
    "eating",
    "eating a pizza"
]

def calculate_clip_text_text_score(prompts_A, prompts_B):

    results = []
    for prompt_A in prompts_A:
        for prompt_B in prompts_B:
            score = metric(prompt_A, prompt_B).item()
            print(f'{prompt_A} {prompt_B} {score}')
            results.append({'prompt_A': prompt_A, 'prompt_B': prompt_B, 'score': score})

    return results

results = calculate_clip_text_text_score(prompts_A, prompts_B)
if results is not None:
    results = pd.DataFrame(results)
    results.to_csv(output_file, index=False)
else:
    print(f'No results for {exp_name} {info}')

# ------------------------------------------------------------------------------------------------

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
output_file = output_file.replace('.csv', '_sentence_transformer.csv')
print('--------------------------------')
print(f'Running sentence transformer for {exp_name} {info}')
print('--------------------------------')

results = []
for prompt_A in prompts_A:
    for prompt_B in prompts_B:
        emb1 = model.encode(prompt_A, convert_to_tensor=True)
        emb2 = model.encode(prompt_B, convert_to_tensor=True)
        score = util.pytorch_cos_sim(emb1, emb2).item()
        print(f'{prompt_A} {prompt_B} {score}')
        results.append({'prompt_A': prompt_A, 'prompt_B': prompt_B, 'score': score})

results = pd.DataFrame(results)
results.to_csv(output_file, index=False)

