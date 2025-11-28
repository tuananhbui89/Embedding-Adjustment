PENGUIN_PROMPTS = [
    "{} in grand canyon", 
    "{} sitting at the beach with a view of the sea",
    "{} in times square",
    "{} wearing sunglasses",
    "{} working on the laptop",
    "{} on a boat in the sea",
    "{} wearing headphones", 
    "{} in a construction outfit",
    "{} on top of a mountain",
    "a koala in the style of {}",
    "a backpack in the style of {}",
    "{} made of crochet",
    "a sweater in the style of {} ",
    "a photo of a {} in Van Gogh style",
]

TSHIRT_PROMPTS = [
    "a laydy wearing {} in grand canyon", 
    "a man wearing {} sitting at the beach with a view of the sea",
    "an older woman wearing {} in times square",
    "a young woman wearing {} wearing sunglasses",
    "a young man wearing {} working on the laptop",
    "a kid wearing {} on a boat in the sea",
    "an older man wearing {} wearing headphones", 
    "a kid wearing {} in a construction outfit",
    "a boy wearing {} on top of a mountain",
    "a koala in the style of {}",
    "a backpack in the style of {}",
    "{} made of crochet",
    "a sweater in the style of {} ",
    "a photo of a {} in Van Gogh style",
]

CLOCK_PROMPTS = [
    "{} in grand canyon", 
    "{} sitting at the beach with a view of the sea",
    "{} in times square",
    "a person wearing sunglasses and holding a {}", # CHANGE THIS
    "a person working on a laptop next to {}", # CHANGE THIS
    "{} on a boat in the sea",
    "{} wearing headphones", 
    "a person wearing a pin striped construction outfit and holding a {}", # CHANGE THIS
    "{} on top of a mountain",
    "a koala in the style of {}",
    "a backpack in the style of {}",
    "{} made of crochet",
    "a sweater in the style of {} ",
    "a photo of a {} in Van Gogh style",
]

RC_CAR_PROMPTS = [
    "{} in grand canyon", 
    "{} sitting at the beach with a view of the sea",
    "{} in times square",
    "a person wearing sunglasses and holding a {}", # CHANGE THIS
    "a person working on a laptop next to {}", # CHANGE THIS
    "{} on a boat in the sea",
    "{} wearing headphones", 
    "a person wearing a pin striped construction outfit and holding a {}", # CHANGE THIS
    "{} on top of a mountain",
    "a koala in the style of {}",
    "a backpack in the style of {}",
    "{} made of crochet",
    "a sweater in the style of {} ",
    "a photo of a {} in Van Gogh style",
]

ORANGES_PROMPTS = [
    "{} in grand canyon", 
    "{} sitting at the beach with a view of the sea",
    "{} in times square",
    "a person wearing sunglasses and holding a {}", # CHANGE THIS
    "a person working on a laptop next to {}", # CHANGE THIS
    "{} on a boat in the sea",
    "{} wearing headphones", 
    "a person wearing a pin striped construction outfit and holding a {}", # CHANGE THIS
    "{} on top of a mountain",
    "a koala in the style of {}",
    "a backpack in the style of {}",
    "{} made of crochet",
    "a sweater in the style of {} ",
    "a photo of a {} in Van Gogh style",
]

settings = {
    "tshirt": {
        "image": "assets/tshirt.jpg",
        "keyword": "this shirt",
        "concept": "a tshirt",
        "target_keyword": "a beige button-up shirt with bold green abstract brushstroke patterns",
        "prompt": "On the beach, a lady sits under a beach umbrella. She's wearing this shirt and has a big smile on her face, with her surfboard hehind her. The sun is setting in the background. The sky is a beautiful shade of orange and purple.",
        "template_prompts": TSHIRT_PROMPTS,
    },
    "penguin": {
        "image": "assets/penguin.jpg",
        "keyword": "this item",
        "concept": "a plushie penguin",
        "target_keyword": "a plushie penguin",
        "prompt": "On Christmas evening, on a crowded sidewalk, this item sits on the road, covered in snow and wearing a Christmas hat.",
        "template_prompts": PENGUIN_PROMPTS,
    },
    "rc_car": {
        "image": "assets/rc_car.jpg",
        "keyword": "this item",
        "concept": "a toy race car",
        "target_keyword": "a colorful toy race car with a smiling driver and chunky wheels",
        "prompt": "A film style shot. On the moon, this item drives across the moon surface. The background is that Earth looms large in the foreground.",
        "template_prompts": RC_CAR_PROMPTS,
    },
    "clock": {
        "image": "assets/clock.jpg",
        "keyword": "this item",
        "concept": "a clock",
        "target_keyword": "a bright yellow twin-bell alarm clock with a bold number “3” on the face",
        "prompt": "In a Bauhaus style room, this item is placed on a shiny glass table, with a vase of flowers next to it. In the afternoon sun, the shadows of the blinds are cast on the wall.",
        "template_prompts": CLOCK_PROMPTS,
    },
    "oranges": {
        "image": "assets/oranges.jpg",
        "keyword": "this item",
        "concept": "a wooden bowl filled with oranges",
        "target_keyword": "a wooden bowl filled with oranges",
        "prompt": "A very close up view of this item. It is placed on a wooden table. The background is a dark room, the TV is on, and the screen is showing a cooking show.",
        "template_prompts": ORANGES_PROMPTS,
    },
}

import os 

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