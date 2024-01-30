import sys
from glob import glob

import numpy as np
from PIL import Image


def prepare_camvid(data_dir, split):
    RGB2ClassName = {(128, 128, 128): "Sky",
                     (64, 0, 64): "Building",
                     (128, 0, 0): "Building",
                     (0, 128, 64): "Building",
                     (64, 192, 0): "Building",
                     (192, 0, 128): "Building",
                     (0, 0, 64): "Pole",
                     (192, 192, 128): "Pole",
                     (192, 0, 64): "Road",
                     (128, 0, 192): "Road",
                     (128, 64, 128): "Road",
                     (0, 0, 192): "Sidewalk",
                     (64, 192, 128): "Sidewalk",
                     (128, 128, 192): "Sidewalk",
                     (128, 128, 0): "Tree",
                     (192, 192, 0): "Tree",
                     (0, 64, 64): "Sign",
                     (128, 128, 64): "Sign",
                     (192, 128, 128): "Sign",
                     (64, 64, 128): "Fence",
                     (64, 0, 128): "Car",
                     (128, 64, 64): "Car",
                     (192, 64, 128): "Car",
                     (64, 128, 192): "Car",
                     (192, 128, 192): "Car",
                     (64, 64, 0): "Pedestrian",
                     (64, 0, 192): "Pedestrian",
                     (64, 128, 64): "Pedestrian",
                     (192, 128, 64): "Pedestrian",
                     (0, 128, 192): "Bicyclist",
                     (192, 0, 192): "Bicyclist",
                     (0, 0, 0): "Void"}

    class_names = ['Sky',
                   'Building',
                   'Pole',
                   'Road',
                   'Sidewalk',
                   'Tree',
                   'Sign',
                   'Fence',
                   'Car',
                   'Pedestrian',
                   'Bicyclist',
                   'Void']

    ClassName2TrainID = {}
    for i, n in enumerate(class_names):
        ClassName2TrainID[n] = i

    mask_files = glob(f"{data_dir}/camvid/annotations/{split}/*.png")

    n = len(mask_files)
    for i, mask_file in enumerate(mask_files):
        print(f"Prepare: {i + 1}|{n}")
        mask = np.array(Image.open(mask_file))
        label = 255 * np.ones(mask.shape[:2], dtype=np.uint8)

        w, h = mask.shape[:2]
        for x in range(w):
            for y in range(h):
                rgb = mask[x, y, :]
                rgb = tuple(rgb)
                if rgb in RGB2ClassName:
                    class_name = RGB2ClassName[rgb]
                else:
                    class_name = 'Void'
                label[x, y] = ClassName2TrainID[class_name]

        label = Image.fromarray(label)
        label_file = mask_file.replace("_L.png", "_labelTrainIds.png")
        label.save(label_file)


if __name__ == "__main__":
    prepare_camvid(sys.argv[1], sys.argv[2])
