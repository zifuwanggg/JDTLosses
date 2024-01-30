import cv2
import torch
import numpy as np

from . import transform

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class BaseDataset(torch.utils.data.Dataset):
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
                 image_prefix=None,
                 image_suffix=None,
                 label_prefix=None,
                 label_suffix=None):
        self.train = train
        self.data_dir = data_dir
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.crop_size = crop_size
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.image_prefix = image_prefix
        self.image_suffix = image_suffix
        self.label_prefix = label_prefix
        self.label_suffix = label_suffix

        self.image_list = None
        self.transform = None
        self.class_names = None
        self.color_map = None


    def __getitem__(self, index):
        image_file = self.image_list[index]
        label_file = self.get_label_file(image_file)

        image_bgr = cv2.imread(image_file, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = np.float32(image_rgb)
        label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)

        if self.reduce_zero_label:
            label[label == 0] = self.ignore_index
            label -= 1
            label[label == (self.ignore_index - 1)] = self.ignore_index

        image, label = self.transform(image, label)

        if self.ignore_index and self.ignore_index != self.num_classes:
            label[label == self.ignore_index] = self.num_classes

        assert image.shape[1:3] == label.shape[0:2]

        return image, label, image_file


    def __len__(self):
        return len(self.image_list)


    def get_label_file(self, image_file):
        if self.image_prefix is not None and self.label_prefix is not None:
            image_file = image_file.replace(self.image_prefix, self.label_prefix)

        if self.image_suffix is not None and self.label_suffix is not None:
            image_file = image_file.replace(self.image_suffix, self.label_suffix)

        return image_file


    def get_image_list(self):
        return


    def get_transform(self):
        mean = [123.68, 116.28, 103.53]
        std = [58.40, 57.12, 57.38]

        if self.train:
            data_transform = transform.Compose([
                transform.RandScale([self.scale_min, self.scale_max]),
                transform.RandomHorizontalFlip(p=0.5),
                transform.Crop([self.crop_size[0], self.crop_size[1]], crop_type="rand", padding=mean, ignore_index=self.ignore_index),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
        else:
            data_transform = transform.Compose([
                transform.Pad([self.crop_size[0], self.crop_size[1]], padding=mean, ignore_index=self.ignore_index),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])

        return data_transform


    def get_class_names(self):
        return


    def get_color_map(self):
        return
