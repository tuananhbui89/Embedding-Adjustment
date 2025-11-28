import os 
import cv2
import matplotlib.pyplot as plt
from PIL import Image

folder_paths = [
    # 'evaluation_massive/omini_subject/clock',
    # 'evaluation_massive/omini_subject/tshirt',
    # 'evaluation_massive/omini_subject/penguin',
    # 'evaluation_massive/omini_subject/rc_car',
    # 'evaluation_massive/omini_subject/oranges',
    # 'evaluation_massive/omini_subject_tea/clock_0.2_0.5',
    # 'evaluation_massive/omini_subject_tea/tshirt_0.2_0.5',
    # 'evaluation_massive/omini_subject_tea/penguin_0.2_0.5',
    # 'evaluation_massive/omini_subject_tea/rc_car_0.2_0.5',
    # 'evaluation_massive/omini_subject_tea/oranges_0.2_0.5',
    'evaluation_massive/omini_subject_tea/clock_0.3_0.5',
    'evaluation_massive/omini_subject_tea/tshirt_0.3_0.5',
    'evaluation_massive/omini_subject_tea/penguin_0.3_0.5',
    'evaluation_massive/omini_subject_tea/rc_car_0.3_0.5',
    'evaluation_massive/omini_subject_tea/oranges_0.3_0.5',

    'evaluation_massive/omini_subject_tea/clock_0.5_0.5',
    'evaluation_massive/omini_subject_tea/tshirt_0.5_0.5',
    'evaluation_massive/omini_subject_tea/penguin_0.5_0.5',
    'evaluation_massive/omini_subject_tea/rc_car_0.5_0.5',
    'evaluation_massive/omini_subject_tea/oranges_0.5_0.5',

    'evaluation_massive/omini_subject_tea_v2/clock_0.3_0.5',
    'evaluation_massive/omini_subject_tea_v2/tshirt_0.3_0.5',
    'evaluation_massive/omini_subject_tea_v2/penguin_0.3_0.5',
    'evaluation_massive/omini_subject_tea_v2/rc_car_0.3_0.5',
    'evaluation_massive/omini_subject_tea_v2/oranges_0.3_0.5',
]

# read all images in the folder, split an image into left and right image, then save the right image only to a new folder with the same name as the original folder
# for folder_path in folder_paths:
#     new_folder_path = folder_path + '_right'
#     os.makedirs(new_folder_path, exist_ok=True)
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith('.jpg') or file_name.endswith('.png'):
#             image = cv2.imread(os.path.join(folder_path, file_name))
#             left_image = image[:, :image.shape[1]//2]
#             right_image = image[:, image.shape[1]//2:]
#             cv2.imwrite(os.path.join(new_folder_path, file_name), right_image)

# read all images in the folder, each image has specific name "index_n_idx.png", for each prompt_index, read all images with the same prompt_index and plot a grid of 10x5 images
  
prompt_indexes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]


for folder_path in folder_paths:
    new_folder_path = f'demo_images/{folder_path.replace("evaluation_massive/", "")}'
    os.makedirs(new_folder_path, exist_ok=True)
    for prompt_index in prompt_indexes:
        fig, axs = plt.subplots(5, 10, figsize=(10, 20))
        for j in range(10):
            for i in range(5):
                n_idx = j * 5 + i
                image = plt.imread(os.path.join(folder_path, f'{prompt_index}_{n_idx}.png'))
                axs[i, j].imshow(image)
                axs[i, j].axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=-0.94)
        plt.savefig(os.path.join(new_folder_path, f'{prompt_index}.jpg'), bbox_inches='tight', pad_inches=0)
        plt.close()