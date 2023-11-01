from glob import glob

import numpy as np
from PIL import Image

data_dir = ""

print(f"Begin processing Mapillary")

train_label_list = glob(f"{data_dir}/mapillary/training/v1.2/labels/*.png")
val_label_list = glob(f"{data_dir}/mapillary/validation/v1.2/labels/*.png")
label_list = train_label_list + val_label_list

length = len(label_list)
for i, label_file in enumerate(label_list):
    print(f"Encoding: {i + 1}|{length}")
    
    label = np.array(Image.open(label_file))
    dst_label_file = label_file.replace(".png", "_labelTrainIds.png")
    Image.fromarray(label.astype(dtype=np.uint8)).save(dst_label_file, "PNG")

print(f"Finish processing Mapillary")