from .base_dataset import BaseDataset


class VOC(BaseDataset):
    def __init__(self,
                 train=True,
                 data_dir="/Users/whoami/datasets",
                 in_channels=3,
                 num_classes=21,
                 scale_min=0.5,
                 scale_max=2.0,
                 crop_size=[512, 512],
                 ignore_index=255,
                 reduce_zero_label=False,
                 reduce_panoptic_zero_label=False,
                 image_prefix="/JPEGImages/",
                 image_suffix=".jpg",
                 label_prefix="/SegmentationClassTrainAug/",
                 label_suffix=".png",
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
                         reduce_panoptic_zero_label=reduce_panoptic_zero_label,
                         image_prefix=image_prefix,
                         image_suffix=image_suffix,
                         label_prefix=label_prefix,
                         label_suffix=label_suffix)


    def get_image_list(self):
        trainaug_txt = f"{self.data_dir}/VOCdevkit/VOC2012/ImageSets/Segmentation/trainaug.txt"
        val_txt = f"{self.data_dir}/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"

        with open(trainaug_txt) as f:
            trainaug_idx = f.readlines()
            trainaug_idx = [line.rstrip('\n') for line in trainaug_idx]

        with open(val_txt) as f:
            val_idx = f.readlines()
            val_idx = [line.rstrip('\n') for line in val_idx]

        train_image_list = [f"{self.data_dir}/VOCdevkit/VOC2012/JPEGImages/{idx}.jpg" for idx in trainaug_idx]
        val_image_list = [f"{self.data_dir}/VOCdevkit/VOC2012/JPEGImages/{idx}.jpg" for idx in val_idx]

        assert len(train_image_list) == 10582, \
            f"`len(train_image_list)`: does not equal to 10582"

        assert len(val_image_list) == 1449, \
            f"`len(val_image_list)`: {len(val_image_list)} does not equal to 1449"

        if self.train:
            return train_image_list
        else:
            return val_image_list


    def get_class_names(self):
        class_names = ["Background", "Aeroplane", "Bicycle", "Bird", "Boat",
                       "Bottle", "Bus", "Car", "Cat", "Chair",
                       "Cow", "Dining Table", "Dog", "Horse", "Motorbike",
                       "Person", "Potted Plant", "Sheep", "Sofa", "Train",
                       "TV Monitor", "Void"]

        return class_names


    def get_color_map(self):
        color_map = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                     [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                     [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
                     [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                     [0, 64, 128], [255, 255, 255]]

        return color_map
