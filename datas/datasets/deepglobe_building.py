from glob import glob

from .base_dataset import BaseDataset


class DeepGlobeBuilding(BaseDataset):
    def __init__(self,
                 train=True,
                 data_dir="/Users/whoami/datasets",
                 fold=0,
                 in_channels=3,
                 num_classes=2,
                 scale_min=0.5,
                 scale_max=2.0,
                 crop_size=[512, 512],
                 ignore_index=2,
                 reduce_zero_label=False,
                 image_prefix=None,
                 image_suffix="sat.jpg",
                 label_prefix=None,
                 label_suffix="label.png",
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

        self.fold = fold
        self.image_list = self.get_image_list()
        self.transform = self.get_transform()
        self.class_names = self.get_class_names()
        self.color_map = self.get_color_map()


    def get_image_list(self):
        train_image_list, val_image_list = [], []

        for city in ["Vegas", "Paris", "Shanghai", "Khartoum"]:
            image_list = glob(f"{self.data_dir}/building/train/{city}_*_sat.jpg")
            image_list.sort()

            fold_size = len(image_list) // 5
            train_image_list_city = image_list[:fold_size * self.fold] + \
                image_list[fold_size * (self.fold + 1):]
            val_image_list_city = image_list[fold_size * self.fold : fold_size * (self.fold + 1)]

            train_image_list.extend(train_image_list_city)
            val_image_list.extend(val_image_list_city)

            if city == "Vegas":
                assert image_list[0] == f"{self.data_dir}/building/train/Vegas_1000_sat.jpg"
                assert len(train_image_list_city) == 3081
                assert len(val_image_list_city) == 770
            elif city == "Paris":
                assert image_list[0] == f"{self.data_dir}/building/train/Paris_1000_sat.jpg"
                assert len(train_image_list_city) == 919
                assert len(val_image_list_city) == 229
            elif city == "Shanghai":
                assert image_list[0] == f"{self.data_dir}/building/train/Shanghai_1001_sat.jpg"
                assert len(train_image_list_city) == 3666
                assert len(val_image_list_city) == 916
            elif city == "Khartoum":
                assert image_list[0] == f"{self.data_dir}/building/train/Khartoum_1001_sat.jpg"
                assert len(train_image_list_city) == 810
                assert len(val_image_list_city) == 202

        assert len(train_image_list) == 8476
        assert len(val_image_list) == 2117

        if self.train:
            return train_image_list
        else:
            return val_image_list


    def get_class_names(self):
        class_names = ["Background", "Building"]

        return class_names


    def get_color_map(self):
        color_map = [[0, 0, 0], [255, 255, 255]]

        return color_map
