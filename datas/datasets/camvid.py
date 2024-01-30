from glob import glob

from .base_dataset import BaseDataset


class CamVid(BaseDataset):
    def __init__(self,
                train=True,
                data_dir="/Users/whoami/datasets",
                in_channels=3,
                num_classes=11,
                scale_min=0.5,
                scale_max=2.0,
                crop_size=[512, 512],
                ignore_index=11,
                reduce_zero_label=False,
                image_prefix="/images/",
                image_suffix=".png",
                label_prefix="/annotations/",
                label_suffix="_labelTrainIds.png",
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
        train_image_list = glob(f"{self.data_dir}/camvid/images/train/*.png")
        val_image_list = glob(f"{self.data_dir}/camvid/images/test/*.png")

        assert len(train_image_list) == 469
        assert len(val_image_list) == 232

        if self.train:
            return train_image_list
        else:
            return val_image_list


    def get_class_names(self):
        class_names = ['Sky', 'Building', 'Pole', 'Road', 'Sidewalk',
                       'Tree', 'Sign', 'Fence', 'Car', 'Pedestrian',
                       'Bicyclist', 'Void']

        return class_names


    def get_color_map(self):
        color_map = [(128, 128, 128), (128, 0, 0), (192, 192, 128), (128, 64, 128), (0, 0, 192),
                     (128, 128, 0), (192, 128, 128), (64, 64, 128), (64, 0, 128), (64, 64, 0),
                     (0, 128, 192), (0, 0, 0)]

        return color_map
