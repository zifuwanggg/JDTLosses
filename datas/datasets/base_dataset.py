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
                 reduce_panoptic_zero_label=False,
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
        self.reduce_panoptic_zero_label = reduce_panoptic_zero_label
        self.image_prefix = image_prefix
        self.image_suffix = image_suffix
        self.label_prefix = label_prefix
        self.label_suffix = label_suffix

        self.image_list = self.get_image_list()
        self.annos, self.anno_map = self.get_annos()
        self.transform = self.get_transform()
        self.class_names = self.get_class_names()
        self.color_map = self.get_color_map()


    def __getitem__(self, index):
        image_file = self.image_list[index]
        label_file = self.get_label_file(image_file)

        image_bgr = cv2.imread(image_file, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = np.float32(image_rgb)
        label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)

        if self.annos is not None:
            panoptic_file = self.get_panoptic_file(label_file)
            panoptic_bgr = cv2.imread(panoptic_file, cv2.IMREAD_COLOR)
            panoptic_rgb = cv2.cvtColor(panoptic_bgr, cv2.COLOR_BGR2RGB)
            panoptic = self.rgb2id(panoptic_rgb)
        else:
            panoptic = None

        if self.reduce_zero_label:
            label[label == 0] = self.ignore_index
            label -= 1
            label[label == (self.ignore_index - 1)] = self.ignore_index

        image, label, panoptic = self.transform(image, label, panoptic)

        if self.ignore_index and self.ignore_index != self.num_classes:
            label[label == self.ignore_index] = self.num_classes

        assert image.shape[1:3] == label.shape[0:2], \
            f"`image.shape[1:3]`: {image.shape[1:3]} does not equal to `label.shape[0:2]`: {label.shape[0:2]}"

        if panoptic is not None:
            assert label.shape == panoptic.shape, \
            "`label.shape`: {label.shape} does not equal to `panoptic.shape`: {panoptic.shape}"
        else:
            panoptic = panoptic_file = image_file

        return image, label, panoptic, image_file, panoptic_file


    def __len__(self):
        return len(self.image_list)


    def get_label_file(self, image_file):
        if self.image_prefix is not None and self.label_prefix is not None:
            image_file = image_file.replace(self.image_prefix, self.label_prefix)

        if self.image_suffix is not None and self.label_suffix is not None:
            image_file = image_file.replace(self.image_suffix, self.label_suffix)

        return image_file


    def get_panoptic_file(self, label_file):
        return


    def rgb2id(self, rgb):
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.int32)

        return rgb[:, :, 0] + 256 * rgb[:, :, 1] + 256 * 256 * rgb[:, :, 2]


    def get_image_list(self):
        return


    def get_annos(self):
        return None, None


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
