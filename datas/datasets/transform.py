import math
import random

import cv2
import torch
import numpy as np

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)

        return image, label


class RandScale(object):
    def __init__(self, scale, aspect_ratio=None, float_label=False):
        self.scale = scale
        self.aspect_ratio = aspect_ratio
        self.float_label = float_label

    def __call__(self, image, label):
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0

        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)

        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio

        image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)

        if self.float_label:
            label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        else:
            label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)

        return image, label


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)

        return image, label


class Pad(object):
    def __init__(self, size, padding, ignore_index):
        self.size = size
        self.padding = padding
        self.ignore_index = ignore_index

    def __call__(self, image, label):
        h, w = label.shape
        pad_h = self.size[0] - h
        pad_w = self.size[1] - w

        if pad_h > 0 or pad_w > 0:
            pad_h = max(pad_h, 0)
            pad_w = max(pad_w, 0)

            pad_h_half = int(pad_h / 2)
            pad_w_half = int(pad_w / 2)

            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.padding)
            label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.ignore_index)

        return image, label


class Crop(object):
    def __init__(self, size, padding, ignore_index, crop_type="center"):
        self.pad = Pad(size, padding, ignore_index)
        self.crop = size
        self.crop_type = crop_type

    def __call__(self, image, label):
        image, label = self.pad(image, label)

        h, w = label.shape

        if self.crop_type == "rand":
            h_off = random.randint(0, h - self.crop[0])
            w_off = random.randint(0, w - self.crop[1])
        else:
            h_off = int((h - self.crop[0]) / 2)
            w_off = int((w - self.crop[1]) / 2)

        image = image[h_off : h_off + self.crop[0], w_off : w_off + self.crop[1]]
        label = label[h_off : h_off + self.crop[0], w_off : w_off + self.crop[1]]

        return image, label


class ToTensor(object):
    def __init__(self, float_label=False):
        self.float_label = float_label

    def __call__(self, image, label):
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        label = torch.from_numpy(label)

        if not isinstance(image, torch.FloatTensor):
            image = image.float()

        if not isinstance(label, torch.LongTensor) and not self.float_label:
            label = label.long()

        return image, label


class Normalize(object):
    def __init__(self, mean, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        if self.std is None:
            for t, m in zip(image, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)

        return image, label


class WindowStandardize(object):
    def __init__(self, window_low, window_high):
        self.window_low = window_low
        self.window_high = window_high

    def __call__(self, image, label):
        image = torch.clamp(image, min=self.window_low, max=self.window_high)
        image = 2 * (image - self.window_low) / (self.window_high - self.window_low) - 1

        return image, label
