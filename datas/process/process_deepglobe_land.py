from glob import glob

import cv2
import numpy as np


data_dir = ""

print(f"Begin processing Land")

class_rgb_values = [[0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]

mask_list = glob(f"{data_dir}/land/train/*_mask.png")

length = len(mask_list)
for i, mask_file in enumerate(mask_list):
    print(f"Encoding: {i + 1}|{length}")
    
    mask = cv2.imread(mask_file, cv2.IMREAD_COLOR)
    label = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    label_file = mask_file.replace("_mask.png", "_label.png") 

    for j, rgb in enumerate(class_rgb_values):
        bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.uint8)
        loc = (mask == bgr).all(axis=-1)
        label[loc] = j
    
    cv2.imwrite(label_file, label)

print(f"Finish processing Land")