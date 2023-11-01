import os
from glob import glob

import numpy as np

from .medical_dataset import MedicalDataset


class KiTS(MedicalDataset):
    def __init__(self,
                 train=True,
                 data_dir="/Users/whoami/datasets",
                 fold=0,
                 num_cases=210,
                 in_channels=1,
                 num_classes=2,
                 scale_min=0.5,
                 scale_max=2.0,
                 window_low=-200,
                 window_high=300,
                 crop_size=[384, 384],
                 ignore_index=255,
                 weighted=False,
                 reduce_zero_label=False,
                 reduce_panoptic_zero_label=False,
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
                         reduce_panoptic_zero_label=reduce_panoptic_zero_label,
                         image_prefix=image_prefix,
                         image_suffix=image_suffix,
                         label_prefix=label_prefix,
                         label_suffix=label_suffix)


    def __getitem__(self, index):
        image_file = self.image_list[index]

        npz = np.load(image_file, allow_pickle=True)
        image = npz.get("ct")
        image = np.float32(image)
        label = npz.get("mask")

        image, label, _ = self.transform(image, label)

        # 0 -> 0, 1 -> 0, 2 -> 1, 255 -> 127
        label >>= 1
        label[label > 1] = 2

        assert image.shape[1:3] == label.shape[0:2], \
            f"`image.shape[1:3]`: {image.shape[1:3]} does not equal to `label.shape[0:2]`: {label.shape[0:2]}"

        return image, label


    def get_image_list(self):
        data_dir = os.path.join(self.data_dir)

        train_image_list, val_image_list = [], []

        cases = list(range(self.num_cases))
        fold_size = self.num_cases // 5
        if self.fold <= 3:
            val_cases = set(cases[fold_size * self.fold : fold_size * (self.fold + 1)])
        else:
            val_cases = set(cases[fold_size * self.fold : ])

        tumor_slices = glob(f"{data_dir}/train/*.npz")

        for slice in tumor_slices:
            case = int(slice.split("/")[-1].split("_")[0])
            if case in val_cases:
                val_image_list.append(slice)
            else:
                train_image_list.append(slice)

        assert len(train_image_list) + len(val_image_list) == 5712, \
            f"`len(train_image_list)`: {len(train_image_list)}, `len(val_image_list)`: {len(val_image_list)}"

        if self.fold == 0:
            assert len(train_image_list) == 4760, \
                f"`len(train_image_list)`: {len(train_image_list)} does not equal to 4760"

            assert len(val_image_list) == 952, \
                f"`len(val_image_list)`: {len(val_image_list)} does not equal to 952"
        elif self.fold == 1:
            assert len(train_image_list) == 4457, \
                f"`len(train_image_list)`: {len(train_image_list)} does not equal to 4457"

            assert len(val_image_list) == 1255, \
                f"`len(val_image_list)`: {len(val_image_list)} does not equal to 1255"
        elif self.fold == 2:
            assert len(train_image_list) == 4294, \
                f"`len(train_image_list)`: {len(train_image_list)} does not equal to 4294"

            assert len(val_image_list) == 1418, \
                f"`len(val_image_list)`: {len(val_image_list)} does not equal to 1418"
        elif self.fold == 3:
            assert len(train_image_list) == 4241, \
                f"`len(train_image_list)`: {len(train_image_list)} does not equal to 4241"

            assert len(val_image_list) == 1471, \
                f"`len(val_image_list)`: {len(val_image_list)} does not equal to 1471"
        elif self.fold == 4:
            assert len(train_image_list) == 5096, \
                f"`len(train_image_list)`: {len(train_image_list)} does not equal to 5096"

            assert len(val_image_list) == 616, \
                f"`len(val_image_list)`: {len(val_image_list)} does not equal to 616"

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
