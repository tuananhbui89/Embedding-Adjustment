import numpy as np 
import torch
from PIL import Image
import random
# import cv2
import os

rare_tokens = [
    "<piie>",
    "<pooli>",
    "<toony>",
    "<bta>",
    "<eiip>",
    "<iloop>",
    "<ynoot>",
    "<atb>"
]

def save_images(images, save_path):
    assert(torch.is_tensor(images))
    assert(len(images.shape) == 4)
    assert(images.shape[1] == 3)
    assert(torch.min(images) == -1)
    assert(torch.max(images) == 1)

    for id, img_pixel in enumerate(images):
        save_path_id = save_path + str(id) + ".png"
        Image.fromarray(
            (img_pixel * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            ).save(save_path_id)

        # Using cv2 will cause color shift
        # cv2.imwrite(save_path_id, (img_pixel * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy())

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

# class logger that create a dictionary to store the logs and print the logs
class Logger:
    def __init__(self):
        self.log_dict = {}

    def log(self, key, value):
        if key not in self.log_dict:    
            self.log_dict[key] = []
        self.log_dict[key].append(value)
        print(f"{key}: {value}")

    def print_log(self):
        for key, value in self.log_dict.items():
            print(f"{key}: {value}")

    def save_log(self, save_path):
        torch.save(self.log_dict, save_path)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def read_prompt_file(prompt_file):
    assert os.path.exists(prompt_file), f"Prompt file {prompt_file} does not exist"
    assert prompt_file.endswith(".csv"), f"Prompt file {prompt_file} is not a csv file"

    # if there is a header, read from the prompt column only
    with open(prompt_file, "r") as f:
        prompts = f.readlines()
    if prompts[0].startswith("prompt"):
        prompts = [line.split(",")[0] for line in prompts[1:]]
    else:
        prompts = [line.strip() for line in prompts]
    
    # assert '{}' in prompts[0], f"Prompt file {prompt_file} does not contain {{}}"

    for i, prompt in enumerate(prompts):
        print('read_prompt_file', i, prompt)

    return prompts

def test_read_prompt_file():
    prompt_file = "prompts/gen_prompt_cs101_barn.csv"
    prompts = read_prompt_file(prompt_file)
    prompts = [prompt.format("<barn>") for prompt in prompts]
    print(prompts)

    prompt_file = "prompts/one_body_prompts_gen.csv"
    prompts = read_prompt_file(prompt_file)
    prompts = [prompt.format("<barn>") for prompt in prompts]
    print(prompts)

if __name__ == "__main__":
    test_read_prompt_file()
