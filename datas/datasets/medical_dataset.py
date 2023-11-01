from . import transform
from .base_dataset import BaseDataset


class MedicalDataset(BaseDataset):
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
                 crop_size=[512, 512],
                 ignore_index=255,
                 reduce_zero_label=False,
                 reduce_panoptic_zero_label=False,
                 float_label=False,
                 image_prefix=None,
                 image_suffix=None,
                 label_prefix=None,
                 label_suffix=None):

        self.fold = fold
        self.num_cases = num_cases
        self.window_low = window_low
        self.window_high = window_high
        self.float_label = float_label

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


    def get_transform(self):
        if self.train:
            data_transform = transform.Compose([
                transform.RandScale([self.scale_min, self.scale_max], weighted=self.weighted),
                transform.RandomHorizontalFlip(p=0.5),
                transform.Crop([self.crop_size[0], self.crop_size[1]], crop_type="rand", padding=self.window_low, ignore_index=self.ignore_index),
                transform.ToTensor(weighted=self.weighted),
                transform.WindowStandardize(low=self.window_low, high=self.window_high)])
        else:
            data_transform = transform.Compose([
                transform.Crop([self.crop_size[0], self.crop_size[1]], crop_type="center", padding=self.window_low, ignore_index=self.ignore_index),
                transform.ToTensor(),
                transform.WindowStandardize(low=self.window_low, high=self.window_high)])

        return data_transform
