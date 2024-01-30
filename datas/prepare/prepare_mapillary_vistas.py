import sys
from glob import glob

import numpy as np
from PIL import Image


def prepare_mapillary_vistas(data_dir):
    train_label_files = glob(f"{data_dir}/mapillary/training/v1.2/labels/*.png")
    val_label_files = glob(f"{data_dir}/mapillary/validation/v1.2/labels/*.png")
    label_files = train_label_files + val_label_files

    n = len(label_files)
    for i, label_file in enumerate(label_files):
        print(f"Prepare: {i + 1}|{n}")

        label = np.array(Image.open(label_file))
        dst_label_file = label_file.replace(".png", "_labelTrainIds.png")
        Image.fromarray(label.astype(dtype=np.uint8)).save(dst_label_file, "PNG")


if __name__ == "__main__":
    prepare_mapillary_vistas(sys.argv[1])
