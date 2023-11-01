from glob import glob

import cv2


data_dir = ""

print(f"Begin processing Road")

mask_list = glob(f"{data_dir}/road/train/*_mask.png")

length = len(mask_list)
for i, mask_file in enumerate(mask_list):
    print(f"Encoding: {i+1}|{length}")
    
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)    
    label = mask / 255
    label_file = mask_file.replace("_mask.png", "_label.png") 
    
    cv2.imwrite(label_file, label)

print(f"Finish processing Road")