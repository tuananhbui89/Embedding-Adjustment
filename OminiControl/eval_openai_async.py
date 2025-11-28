import os
import json
import time
import re
import csv
import argparse
import base64
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from openai import OpenAI
from openai import AzureOpenAI
import asyncio
from openai import AsyncAzureOpenAI

from utils import read_prompt_file
import pandas as pd

def ensure_dir(directory):
    """Make sure the directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def load_image(image_path):
    """Load an image from path."""
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    return image


def encode_image_into_base64(image):
    """Convert PIL image to base64."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def read_prompt_template(file_path):
    """Read prompt template from file."""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: Could not find prompt file: {file_path}")
        return None


async def call_openai_async(reference_image_base64, image_base64, system_prompt, gen_prompt, model="gpt-4o-mini", max_retries=3):
    """Async version of call_openai."""
    client = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01",
        azure_endpoint=f"https://pg-llm.openai.azure.com/openai/deployments/{model}/chat/completions?api-version=2024-08-01-preview"
    )
    
    for attempt in range(max_retries):
        try:
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": system_prompt},
                        {"type": "text", "text": "Here is reference image A: "},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{reference_image_base64}", "detail": "high"}},
                        {"type": "text", "text": "Here is generated image O: "},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "high"}},
                        {"type": "text", "text": f"Here is the prompt P: {gen_prompt}"}
                    ]
                }
            ]
            
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=300
            )
            
            return completion.choices[0].message.content
        
        except Exception as e:
            print(f"Error calling OpenAI API: {e}. Retrying... ({attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
    
    return None

async def evaluate_image_async(anchor_path, image_path, system_prompt, prompt):
    """Async version of evaluate_image_vs_image."""
    reference_image = load_image(anchor_path)
    target_image = load_image(image_path)
    
    reference_image_base64 = encode_image_into_base64(reference_image)
    target_image_base64 = encode_image_into_base64(target_image)
    
    content = await call_openai_async(reference_image_base64, target_image_base64, system_prompt, prompt)
    
    if content:
        prompt_score_pattern = r"PromptScore:\s*(\d+)"
        subject_score_pattern = r"SubjectScore:\s*(\d+)"
        
        prompt_score_matches = re.findall(prompt_score_pattern, content, re.IGNORECASE)
        subject_score_matches = re.findall(subject_score_pattern, content, re.IGNORECASE)
        
        if prompt_score_matches and subject_score_matches:
            prompt_score = int(prompt_score_matches[0])
            subject_score = int(subject_score_matches[0])
            
            prompt_score = min(max(prompt_score, 0), 4)
            subject_score = min(max(subject_score, 0), 4)
            
            return {
                "filename": image_path,
                "promptscore": prompt_score,
                "subjectscore": subject_score,
                "content": content,
                "prompt": prompt
            }
    
    return None

def save_results(all_results, output_file):
    """Save evaluation results to files."""
    # Calculate overall stats
    if all_results:

        prompt_scores = [item['promptscore'] for item in all_results]
        subject_scores = [item['subjectscore'] for item in all_results]
        avg_prompt_score = sum(prompt_scores) / len(prompt_scores)
        avg_subject_score = sum(subject_scores) / len(subject_scores)
        
        prompt_score_distribution = {str(score): prompt_scores.count(score) for score in range(5)}
        subject_score_distribution = {str(score): subject_scores.count(score) for score in range(5)}
        
        # Save results in CSV format (sorted by filename)
        with open(output_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "promptscore", "subjectscore", "content", "prompt"])
            for result in all_results:
                # Clean content by removing special characters
                cleaned_content = result["content"].replace('```', '').replace('\n', ' ').strip()
                writer.writerow([result["filename"], result["promptscore"], result["subjectscore"], cleaned_content, result["prompt"]])
        
        # Save summary in CSV format
        with open(output_file.replace('.csv', '_summary.csv'), "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["total_images", "average_promptscore", "average_subjectscore"])
            writer.writerow([
                len(prompt_scores),
                f"{avg_prompt_score:.2f}",
                f"{avg_subject_score:.2f}"
            ])
        
        print("\nOverall Image Comparison Results")
        print(f"1. Total images evaluated: {len(prompt_scores)}")
        print(f"2. Average PromptScore: {avg_prompt_score:.2f}")
        print(f"3. Average SubjectScore: {avg_subject_score:.2f}")
        print(f"4. PromptScore distribution: {prompt_score_distribution}")
        print(f"5. SubjectScore distribution: {subject_score_distribution}")
        
        return {
            "total_images_evaluated": len(prompt_scores),
            "average_promptscore": avg_prompt_score,
            "average_subjectscore": avg_subject_score,
            "promptscore_distribution": prompt_score_distribution,
            "subjectscore_distribution": subject_score_distribution
        }
    
    return None


async def main_async(args):
    """Async main function."""
    # ... [load prompts and setup] ...

    df = pd.read_csv(args.prompt_file)
    prompts = df.iloc[:, 0].tolist()

    system_prompt = read_prompt_template(args.system_prompt_file)
    ensure_dir(args.output_dir)

    print("\nPrompts to evaluate:")
    for prompt in prompts:
        print(f"  - {prompt}")
    
    tasks = []
    for prompt_id, prompt in enumerate(prompts):
        for image_id in range(args.num_images):
            image_path = os.path.join(args.images_folder, f'{prompt_id}_{image_id}.png')
            if os.path.exists(image_path):
                tasks.append(evaluate_image_async(args.anchor_image_path, image_path, system_prompt, prompt))
    
    # Run all tasks concurrently with progress bar
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating"):
        result = await coro
        if result:
            results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Image Alignment Evaluation using Qwen VL",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Required arguments
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--images_folder", 
        type=str, 
        required=True,
        help="Directory containing target/generated images"
    )
    
    parser.add_argument(
        "--prompt_file", 
        type=str, 
        default=None,
        help="Path to prompt template file (default: auto-select based on mode)"
    )

    parser.add_argument(
        "--num_images",
        type=int,
        default=50,
        help="Number of images to evaluate"
    )

    # Mode-specific arguments
    parser.add_argument(
        "--anchor_image_path", 
        type=str, 
        default=None,
        help="Path to anchor image"
    )

    parser.add_argument(
        "--eval_model",
        type=str,
        default="qwen2.5vl:32b",
        help="Model to use"
    )

    parser.add_argument(
        "--info",
        type=str,
        default="",
        help="Info to add to the output file"
    )

    parser.add_argument(
        "--system_prompt_file",
        type=str,
        default="eval_system_prompt.txt",
        help="Path to system prompt file"
    )
        
    args = parser.parse_args()

    output_file = os.path.join(args.output_dir, f'{args.eval_model}_evaluation_results_{args.info}.csv')

    ensure_dir(args.output_dir)

    df = pd.read_csv(args.prompt_file)
    prompts = df.iloc[:, 0].tolist()

    print("\nPrompts to evaluate:")
    for prompt in prompts:
        print(f"  - {prompt}")
    print()

    # Run async main
    all_results = asyncio.run(main_async(args))


    # Save all results
    print(f"\nSaving results to {output_file}")
    save_results(all_results, output_file)

if __name__ == "__main__":
    main()
