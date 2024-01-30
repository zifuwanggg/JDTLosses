from .base_dataset import BaseDataset


class PASCALContext(BaseDataset):
    def __init__(self,
                 train=True,
                 data_dir="/Users/whoami/datasets",
                 in_channels=3,
                 num_classes=59,
                 scale_min=0.5,
                 scale_max=2.0,
                 crop_size=[512, 512],
                 ignore_index=255,
                 reduce_zero_label=True,
                 image_prefix="/JPEGImages/",
                 image_suffix=".jpg",
                 label_prefix="/SegmentationClassContext/",
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
                         image_prefix=image_prefix,
                         image_suffix=image_suffix,
                         label_prefix=label_prefix,
                         label_suffix=label_suffix)

        self.image_list = self.get_image_list()
        self.transform = self.get_transform()
        self.class_names = self.get_class_names()
        self.color_map = self.get_color_map()


    def get_image_list(self):
        train_txt = f"{self.data_dir}/VOCdevkit/VOC2010/ImageSets/SegmentationContext/train.txt"
        val_txt = f"{self.data_dir}/VOCdevkit/VOC2010/ImageSets/SegmentationContext/val.txt"

        with open(train_txt) as f:
            train_idx = f.readlines()
            train_idx = [line.rstrip('\n') for line in train_idx]

        with open(val_txt) as f:
            val_idx = f.readlines()
            val_idx = [line.rstrip('\n') for line in val_idx]

        train_image_list = [f"{self.data_dir}/VOCdevkit/VOC2010/JPEGImages/{idx}.jpg" for idx in train_idx]
        val_image_list = [f"{self.data_dir}/VOCdevkit/VOC2010/JPEGImages/{idx}.jpg" for idx in val_idx]

        assert len(train_image_list) == 4996
        assert len(val_image_list) == 5104

        if self.train:
            return train_image_list
        else:
            return val_image_list


    def get_class_names(self):
        class_names = ["Aeroplane", "Bag", "Bed", "Bedclothes", "Bench",
                       "Bicycle", "Bird", "Boat", "Book", "Bottle",
                       "Building", "Bus", "Cabinet", "Car", "Cat",
                       "Ceiling", "Chair", "Cloth", "Computer", "Cow",
                       "Cup", "Curtain", "Dog", "Door", "Fence",
                       "Floor", "Flower", "Food", "Grass", "Ground",
                       "Horse", "Keyboard", "Light", "Motorbike", "Mountain",
                       "Mouse", "Person", "Plate", "Platform", "Potted Plant",
                       "Road", "Rock", "Sheep", "Shelves", "Sidewalk",
                       "Sign", "Sky", "Snow", "Sofa", "Table",
                       "Track", "Train", "Tree", "Truck", "TV Monitor",
                       "Wall", "Water", "Window", "Wood", "Void"]

        return class_names


    def get_color_map(self):
        color_map = [[180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3], [120, 120, 80],
                     [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7], [224, 5, 255],
                     [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                     [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3], [0, 102, 200],
                     [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71], [255, 9, 224],
                     [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255], [8, 255, 214],
                     [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                     [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7], [255, 122, 8],
                     [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255], [235, 12, 255],
                     [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15], [20, 255, 0],
                     [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                     [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255], [120, 120, 120]]

        return color_map
