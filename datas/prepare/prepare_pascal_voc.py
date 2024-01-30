import os
import sys
from glob import glob
from shutil import copyfile

import numpy as np
from PIL import Image


def prepare_pascal_voc(data_dir):
    label_list = glob(os.path.join(data_dir, "VOCdevkit/VOC2012/SegmentationClass/*.png"))
    aug_label_list = glob(os.path.join(data_dir, "VOCdevkit/VOC2012/SegmentationClassAug/*.png"))
    dst_label_dir = os.path.join(data_dir, "VOCdevkit/VOC2012/SegmentationClassTrainAug")
    os.makedirs(dst_label_dir, exist_ok=True)

    aug_idx_set = set()
    n = len(aug_label_list)
    for i, label_file in enumerate(aug_label_list):
        print(f"Prepare aug label: {i}|{n}")
        aug_idx_set.add(label_file.split("/")[-1])
        dst_label_file = label_file.replace("SegmentationClassAug", "SegmentationClassTrainAug")
        copyfile(label_file, dst_label_file)

    n = len(label_list)
    for i, label_file in enumerate(label_list):
        print(f"Prepare label: {i}|{n}")
        idx = label_file.split("/")[-1]
        if idx not in aug_idx_set:
            label = np.array(Image.open(label_file))
            dst_label_file = label_file.replace("SegmentationClass", "SegmentationClassTrainAug")
            Image.fromarray(label.astype(dtype=np.uint8)).save(dst_label_file, "PNG")


if __name__ == "__main__":
    prepare_pascal_voc(sys.argv[1])
