import json
from glob import glob

from .base_dataset import BaseDataset


class Cityscapes(BaseDataset):
    def __init__(self,
                 train=True,
                 data_dir="/Users/whoami/datasets",
                 in_channels=3,
                 num_classes=19,
                 scale_min=0.5,
                 scale_max=2.0,
                 crop_size=[512, 1024],
                 ignore_index=255,
                 reduce_zero_label=False,
                 reduce_panoptic_zero_label=False,
                 image_prefix="/leftImg8bit/",
                 image_suffix="_leftImg8bit.png",
                 label_prefix="/gtFine/",
                 label_suffix="_gtFine_labelTrainIds.png",
                 **kwargs):

        super().__init__(train=train,
                         data_dir=data_dir,
                         in_channels=in_channels,
                         num_classes=num_classes,
                         scale_min=scale_min,
                         scale_max=scale_max,
                         crop_size=crop_size,
                         ignore_index=ignore_index,
                         reduce_zero_label=reduce_zero_label,
                         reduce_panoptic_zero_label=reduce_panoptic_zero_label,
                         image_prefix=image_prefix,
                         image_suffix=image_suffix,
                         label_prefix=label_prefix,
                         label_suffix=label_suffix)


    def get_panoptic_file(self, label_file):
        split, city = label_file.split("/")[-3 : -1]

        panoptic_file = label_file.replace(f"{split}/{city}", f"cityscapes_panoptic_{split}_trainId")
        panoptic_file = panoptic_file.replace(self.label_suffix, "_gtFine_panoptic.png")

        return panoptic_file


    def get_image_list(self):
        train_image_list = glob(f"{self.data_dir}/cityscapes/leftImg8bit/train/**/*.png", recursive=True)
        val_image_list = glob(f"{self.data_dir}/cityscapes/leftImg8bit/val/**/*.png", recursive=True)

        assert len(train_image_list) == 2975, \
            f"`len(train_image_list)`: {len(train_image_list)} does not equal to 2975"

        assert len(val_image_list) == 500, \
            f"`len(val_image_list)`: {len(val_image_list)} does not equal to 500"

        if self.train:
            return train_image_list
        else:
            return val_image_list


    def get_annos(self):
        split = "train" if self.train else "val"
        json_file = f"{self.data_dir}/cityscapes/gtFine/cityscapes_panoptic_{split}_trainId.json"

        annos = json.load(open(json_file))["annotations"]

        anno_map = {}
        for i, anno in enumerate(annos):
            file_name = anno["file_name"]
            anno_map[file_name] = i

        return annos, anno_map


    def get_class_names(self):
        class_names = ["Road", "Sidewalk", "Building", "Wall", "Fence",
                       "Pole", "Traffic Light", "Traffic Sign", "Vegetation", "Terrain",
                       "Sky", "Person", "Rider", "Car", "Truck",
                       "Bus", "Train", "Motorcycle", "Bicycle", "Void"]

        return class_names


    def get_color_map(self):
        color_map = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
                     [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
                     [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                     [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0]]

        return color_map
