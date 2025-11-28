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


def call_openai(reference_image_base64, image_base64, system_prompt, gen_prompt, model="gpt-4o-mini", max_retries=3):
    """Call Qwen VL model using OpenAI client format."""
    client = AzureOpenAI(
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
            
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=300
            )
            
            return completion.choices[0].message.content
        
        except Exception as e:
            print(f"Error calling OpenAI API: {e}. Retrying... ({attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(1)
    
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

def mode_img_vs_img(reference_image_path, target_image_path, system_prompt, gen_prompt):

    # load and process images
    reference_image = load_image(reference_image_path)
    target_image = load_image(target_image_path)

    # encode images to base64
    reference_image_base64 = encode_image_into_base64(reference_image)
    target_image_base64 = encode_image_into_base64(target_image)

    # evaluate image vs image
    result = evaluate_image_vs_image(reference_image_base64, target_image_base64, target_image_path, system_prompt, gen_prompt)
    return result

def evaluate_image_vs_image(reference_image_base64, target_image_base64, filename, system_prompt, gen_prompt):
    """Evaluate a single image pair and return the result."""
    max_retries = 3
    retry_count = 0
    content = None
    
    while retry_count < max_retries:
        try:
            content = call_openai(reference_image_base64, target_image_base64, system_prompt, gen_prompt)
            if content:
                break
            retry_count += 1
        except Exception as e:
            print(f"Error evaluating {filename}: {e}. Retrying...")
            retry_count += 1
            time.sleep(1)
    
    # If successful, process the result
    if content:
        # Extract PromptScore and SubjectScore
        prompt_score_pattern = r"PromptScore:\s*(\d+)"
        subject_score_pattern = r"SubjectScore:\s*(\d+)"
        
        prompt_score_matches = re.findall(prompt_score_pattern, content, re.IGNORECASE)
        subject_score_matches = re.findall(subject_score_pattern, content, re.IGNORECASE)
        
        if prompt_score_matches and subject_score_matches:
            prompt_score = int(prompt_score_matches[0])
            subject_score = int(subject_score_matches[0])
            
            # Validate scores are 0-4
            if prompt_score not in [0, 1, 2, 3, 4]:
                print(f"Invalid PromptScore for {filename}: {prompt_score}. Should be 0-4.")
                prompt_score = min(max(prompt_score, 0), 4)  # Clamp to 0-4
            
            if subject_score not in [0, 1, 2, 3, 4]:
                print(f"Invalid SubjectScore for {filename}: {subject_score}. Should be 0-4.")
                subject_score = min(max(subject_score, 0), 4)  # Clamp to 0-4
            
            # Store result
            result = {
                "filename": filename,
                "promptscore": prompt_score,
                "subjectscore": subject_score,
                "content": content,
                "prompt": gen_prompt
            }
            
            return result
        else:
            print(f"Failed to extract scores from Qwen VL response for {filename}.")
            print(f"Content: {content}")
    else:
        print(f"Failed to evaluate {filename} after {max_retries} attempts")
    
    return None


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

    system_prompt = read_prompt_template(args.system_prompt_file)

    ensure_dir(args.output_dir)

    df = pd.read_csv(args.prompt_file)
    prompts = df.iloc[:, 0].tolist()

    print("\nPrompts to evaluate:")
    for prompt in prompts:
        print(f"  - {prompt}")
    print()

    all_results = []
    start_time = time.time()

    for prompt_id, prompt in enumerate(prompts):
        for image_id in range(args.num_images):
            image_path = os.path.join(args.images_folder, f'{prompt_id}_{image_id}.png')
            if not os.path.exists(image_path):
                print(f'Image {image_path} does not exist')
                continue
            
            result = mode_img_vs_img(args.anchor_image_path, image_path, system_prompt, prompt)
            
            if result:
                all_results.append(result)
                current_time = time.time()
                print(f"prompt: {prompt}, image: {image_path}, PromptScore: {result['promptscore']}, SubjectScore: {result['subjectscore']}, Time taken: {current_time - start_time} seconds")
    
    # Save all results
    print(f"\nSaving results to {output_file}")
    save_results(all_results, output_file)

if __name__ == "__main__":
    main()
