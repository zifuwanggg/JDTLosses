from glob import glob
import numpy as np

from .medical_dataset import MedicalDataset


class LiTS(MedicalDataset):
    def __init__(self,
                 train=True,
                 data_dir="/Users/whoami/datasets",
                 fold=0,
                 num_cases=131,
                 in_channels=1,
                 num_classes=2,
                 scale_min=0.5,
                 scale_max=2.0,
                 window_low=-60,
                 window_high=140,
                 crop_size=[384, 384],
                 ignore_index=255,
                 weighted=False,
                 reduce_zero_label=False,
                 image_prefix=None,
                 image_suffix=None,
                 label_prefix=None,
                 label_suffix=None,
                 **kwargs):
        super().__init__(train=train,
                         data_dir=data_dir,
                         fold=fold,
                         num_cases=num_cases,
                         in_channels=in_channels,
                         num_classes=num_classes,
                         scale_min=scale_min,
                         scale_max=scale_max,
                         window_low=window_low,
                         window_high=window_high,
                         crop_size=crop_size,
                         ignore_index=ignore_index,
                         weighted=weighted,
                         reduce_zero_label=reduce_zero_label,
                         image_prefix=image_prefix,
                         image_suffix=image_suffix,
                         label_prefix=label_prefix,
                         label_suffix=label_suffix)

        self.image_list = self.get_image_list()
        self.transform = self.get_transform()
        self.class_names = self.get_class_names()
        self.color_map = self.get_color_map()


    def __getitem__(self, index):
        image_file = self.image_list[index]

        npz = np.load(image_file, allow_pickle=True)
        image = npz.get("ct")
        image = np.float32(image)
        label = npz.get("mask")

        image, label = self.transform(image, label)

        # 0 -> 0, 1 -> 0, 2 -> 1, 255 -> 127
        label >>= 1
        label[label > 1] = 2

        assert image.shape[1:3] == label.shape[0:2]

        return image, label, image_file


    def get_image_list(self):
        train_image_list, val_image_list = [], []

        cases = list(range(self.num_cases))
        fold_size = self.num_cases // 5
        if self.fold <= 3:
            val_cases = set(cases[fold_size * self.fold : fold_size * (self.fold + 1)])
        else:
            val_cases = set(cases[fold_size * self.fold : ])

        tumor_slices = glob(f"{self.data_dir}/train/*.npz")

        for slice in tumor_slices:
            case = int(slice.split("/")[-1].split("_")[0])
            if case in val_cases:
                val_image_list.append(slice)
            else:
                train_image_list.append(slice)

        assert len(train_image_list) + len(val_image_list) == 7190

        if self.fold == 0:
            assert len(train_image_list) == 5870
            assert len(val_image_list) == 1320
        elif self.fold == 1:
            assert len(train_image_list) == 6301
            assert len(val_image_list) == 889
        elif self.fold == 2:
            assert len(train_image_list) == 6731
            assert len(val_image_list) == 459
        elif self.fold == 3:
            assert len(train_image_list) == 4912
            assert len(val_image_list) == 2278
        elif self.fold == 4:
            assert len(train_image_list) == 4946
            assert len(val_image_list) == 2244

        if self.train:
            return train_image_list
        else:
            return val_image_list


    def get_class_names(self):
        class_names = ["Background", "Foreground"]

        return class_names


    def get_color_map(self):
        color_map = [[0, 0, 0], [255, 255, 255]]

        return color_map
