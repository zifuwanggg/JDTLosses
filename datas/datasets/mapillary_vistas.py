from glob import glob

from .base_dataset import BaseDataset


class MapillaryVistas(BaseDataset):
    def __init__(self,
                 train=True,
                 data_dir="/Users/whoami/datasets",
                 in_channels=3,
                 num_classes=65,
                 scale_min=0.5,
                 scale_max=2.0,
                 crop_size=[512, 1024],
                 ignore_index=65,
                 reduce_zero_label=False,
                 image_prefix="/images/",
                 image_suffix=".jpg",
                 label_prefix="/v1.2/labels/",
                 label_suffix="_labelTrainIds.png",
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
        train_image_list = glob(f"{self.data_dir}/mapillary/training/images/*.jpg")
        val_image_list = glob(f"{self.data_dir}/mapillary/validation/images/*.jpg")

        assert len(train_image_list) == 18000
        assert len(val_image_list) == 2000

        if self.train:
            return train_image_list
        else:
            return val_image_list


    def get_class_names(self):
        class_names = ["Bird", "Ground Animal", "Curb", "Fence", "Guard Rail",
                       "Barrier", "Wall", "Bike Lane", "Crosswalk - Plain", "Curb Cut",
                       "Parking", "Pedestrian Area", "Rail Track", "Road", "Service Lane",
                       "Sidewalk", "Bridge", "Building", "Tunnel", "Person",
                       "Bicyclist", "Motorcyclist", "Other Rider", "Lane Marking - Crosswalk", "Lane Marking - General",
                       "Mountain", "Sand", "Sky", "Snow", "Terrain",
                       "Vegetation", "Water", "Banner", "Bench", "Bike Rack",
                       "Billboard", "Catch Basin", "CCTV Camera", "Fire Hydrant", "Junction Box",
                       "Mailbox", "Manhole", "Phone Booth", "Pothole", "Street Light",
                       "Pole", "Traffic Sign Frame", "Utility Pole", "Traffic Light", "Traffic Sign (Back)",
                       "Traffic Sign (Front)", "Trash Can", "Bicycle", "Boat", "Bus",
                       "Car", "Caravan", "Motorcycle", "On Rails", "Other Vehicle",
                       "Trailer", "Truck", "Wheeled Slow", "Car Mount", "Ego Vehicle",
                       "Void"]

        return class_names


    def get_color_map(self):
        color_map = [[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153], [180, 165, 180],
                     [90, 120, 150], [102, 102, 156], [128, 64, 255], [140, 140, 200], [170, 170, 170],
                     [250, 170, 160], [96, 96, 96], [230, 150, 140], [128, 64, 128], [110, 110, 110],
                     [244, 35, 232], [150, 100, 100], [70, 70, 70], [150, 120, 90], [220, 20, 60],
                     [255, 0, 0], [255, 0, 100], [255, 0, 200], [200, 128, 128], [255, 255, 255],
                     [64, 170, 64], [230, 160, 50], [70, 130, 180], [190, 255, 255], [152, 251, 152],
                     [107, 142, 35], [0, 170, 30], [255, 255, 128], [250, 0, 30], [100, 140, 180],
                     [220, 220, 220], [220, 128, 128], [222, 40, 40], [100, 170, 30], [40, 40, 40],
                     [33, 33, 33], [100, 128, 160], [142, 0, 0], [70, 100, 150], [210, 170, 100],
                     [153, 153, 153], [128, 128, 128], [0, 0, 80], [250, 170, 30], [192, 192, 192],
                     [220, 220, 0], [140, 140, 20], [119, 11, 32], [150, 0, 255], [0, 60, 100],
                     [0, 0, 142], [0, 0, 90], [0, 0, 230], [0, 80, 100], [128, 64, 64],
                     [0, 0, 110], [0, 0, 70], [0, 0, 192], [32, 32, 32], [120, 10, 10],
                     [0, 0, 0]]

        return color_map
