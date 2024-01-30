import sys
from glob import glob

import cv2


def prepare_deepglobe_road(data_dir):
    mask_list = glob(f"{data_dir}/road/train/*_mask.png")

    n = len(mask_list)
    for i, mask_file in enumerate(mask_list):
        print(f"Prepare: {i+1}|{n}")

        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        label = mask / 255
        label_file = mask_file.replace("_mask.png", "_label.png")

        cv2.imwrite(label_file, label)


if __name__ == "__main__":
    prepare_deepglobe_road(sys.argv[1])
