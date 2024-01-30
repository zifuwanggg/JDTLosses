import numpy as np

from . import transform
from .medical_dataset import MedicalDataset


class QUBIQ(MedicalDataset):
    def __init__(self,
                 train=True,
                 data_dir="/Users/whoami/datasets",
                 qubiq_dataset="brain-growth",
                 qubiq_label="weighted",
                 qubiq_task=0,
                 fold=0,
                 num_cases=39,
                 num_raters=7,
                 in_channels=1,
                 num_classes=2,
                 scale_min=0.5,
                 scale_max=2.0,
                 window_low=0,
                 window_high=255,
                 crop_size=[256, 256],
                 ignore_index=255,
                 reduce_zero_label=False,
                 image_prefix=None,
                 image_suffix=None,
                 label_prefix=None,
                 label_suffix=None):
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
                         reduce_zero_label=reduce_zero_label,
                         image_prefix=image_prefix,
                         image_suffix=image_suffix,
                         label_prefix=label_prefix,
                         label_suffix=label_suffix)

        assert qubiq_dataset in ["qubiq/brain-growth",
                                 "qubiq/brain-tumor",
                                 "qubiq/kidney",
                                 "qubiq/prostate"]
        assert qubiq_label in ["majority", "uniform", "weighted"]

        if qubiq_label in ["majority"]:
            float_label = False
        elif qubiq_label in ["uniform", "weighted"]:
            float_label = True

        self.qubiq_dataset = qubiq_dataset
        self.qubiq_label = qubiq_label
        self.qubiq_task = qubiq_task
        self.num_raters = num_raters
        self.float_label = float_label


    def __getitem__(self, index):
        image_file = self.image_list[index]
        label_file = self.get_label_file(image_file)

        image = np.load(image_file, allow_pickle=True).get("data")
        image = np.float32(image)
        label = np.load(label_file, allow_pickle=True).get("data")

        image, label = self.transform(image, label)

        if self.ignore_index and \
           self.ignore_index != self.num_classes:
            label[label == self.ignore_index] = self.num_classes

        assert image.shape[1:3] == label.shape[0:2]

        return image, label, image_file


    def get_label_file(self, image_file):
        label_file = image_file.replace(
            "image.npz", f"task{self.task}_{self.qubiq_label}_label.npz")

        return label_file


    def get_image_list(self):
        train_image_list, val_image_list = [], []

        fold_size = self.num_cases // 5
        cases = list(range(1, self.num_cases + 1))

        if self.fold <= 3:
            val_cases = set(cases[fold_size * self.fold : fold_size * (self.fold + 1)])
        else:
            val_cases = set(cases[fold_size * self.fold : ])

        for case in range(1, self.num_cases + 1):
            image_path = f"{self.data_dir}/{self.qubiq_dataset}/case{str(case).zfill(2)}/image.npz"
            if case in val_cases:
                val_image_list.append(image_path)
            else:
                train_image_list.append(image_path)

        assert len(val_image_list) == len(val_cases)
        assert len(train_image_list) + len(val_image_list) == len(cases)

        if self.train:
            return train_image_list
        else:
            return val_image_list


    def get_transform(self):
        if self.train:
            data_transform = transform.Compose([
                transform.RandScale([self.scale_min, self.scale_max],
                                    float_label=self.float_label),
                transform.RandomHorizontalFlip(p=0.5),
                transform.Crop([self.crop_size[0], self.crop_size[1]],
                               crop_type="rand",
                               padding=self.window_low,
                               ignore_index=self.ignore_index),
                transform.ToTensor(float_label=self.float_label),
                transform.WindowStandardize(low=self.window_low, high=self.window_high)])
        else:
            data_transform = transform.Compose([
                transform.Crop([self.crop_size[0], self.crop_size[1]],
                               crop_type="center",
                               padding=self.window_low,
                               ignore_index=self.ignore_index),
                transform.ToTensor(float_label=self.float_label),
                transform.WindowStandardize(low=self.window_low, high=self.window_high)])

        return data_transform
