from glob import glob

from .base_dataset import BaseDataset


class DarkZurich(BaseDataset):
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
                 image_prefix="/rgb_anon/",
                 image_suffix="_rgb_anon.png",
                 label_prefix="/gt/",
                 label_suffix="_gt_labelTrainIds.png",
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
                         image_prefix=image_prefix,
                         image_suffix=image_suffix,
                         label_prefix=label_prefix,
                         label_suffix=label_suffix)

        self.image_list = self.get_image_list()
        self.transform = self.get_transform()
        self.class_names = self.get_class_names()
        self.color_map = self.get_color_map()


    def get_image_list(self):
        train_image_list = glob(f"{self.data_dir}/dark_zurich/rgb_anon/val/night/GOPR0356/*.png")
        val_image_list = train_image_list

        assert len(train_image_list) == 50
        assert len(val_image_list) == 50

        if self.train:
            return train_image_list
        else:
            return val_image_list


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
