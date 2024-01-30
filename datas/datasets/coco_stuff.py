from glob import glob

from .base_dataset import BaseDataset


class COCOStuff(BaseDataset):
    def __init__(self,
                 train=True,
                 data_dir="/Users/whoami/datasets",
                 in_channels=3,
                 num_classes=171,
                 scale_min=0.5,
                 scale_max=2.0,
                 crop_size=[512, 512],
                 ignore_index=255,
                 reduce_zero_label=False,
                 image_prefix="/images/",
                 image_suffix=".jpg",
                 label_prefix="/annotations/",
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
        train_image_list = glob(f"{self.data_dir}/coco_stuff164k/images/train2017/*.jpg")
        val_image_list = glob(f"{self.data_dir}/coco_stuff164k/images/val2017/*.jpg")

        assert len(train_image_list) == 118287
        assert len(val_image_list) == 5000

        if self.train:
            return train_image_list
        else:
            return val_image_list


    def get_class_names(self):
        class_names = ["Person", "Bicycle", "Car", "Motorcycle", "Airplane",
                       "Bus", "Train", "Truck", "Boat", "Traffic Light",
                       "Fire Hydrant", "Stop Sign", "Parking Meter", "Bench", "Bird",
                       "Cat", "Dog", "Horse", "Sheep", "Cow",
                       "Elephant", "Bear", "Zebra", "Giraffe", "Backpack",
                       "Umbrella", "Handbag", "Tie", "Suitcase", "Frisbee",
                       "Skis", "Snowboard", "Sports Ball", "Kite", "Baseball Bat",
                       "Baseball Glove", "Skateboard", "Surfboard", "Tennis Racket", "Bottle",
                       "Wine Glass", "Cup", "Fork", "Knife", "Spoon",
                       "Bowl", "Banana", "Apple", "Sandwich", "Orange",
                       "Broccoli", "Carrot", "Hot Dog", "Pizza", "Donut",
                       "Cake", "Chair", "Couch", "Potted Plant", "Bed",
                       "Dining Table", "Toilet", "TV", "Laptop", "Mouse",
                       "Remote", "Keyboard", "Cell Phone", "Microwave", "Oven",
                       "Toaster", "Sink", "Refrigerator", "Book", "Clock",
                       "Vase", "Scissors", "Teddy Bear", "Hair Drier", "Toothbrush",
                       "Banner", "Blanket", "Branch", "Bridge", "Building-other",
                       "Bush", "Cabinet", "Cage", "Cardboard", "Carpet",
                       "Ceiling-other", "Ceiling-tile", "Cloth", "Clothes", "Clouds",
                       "Counter", "Cupboard", "Curtain", "Desk-stuff", "Dirt",
                       "Door-stuff", "Fence", "Floor-marble", "Floor-other", "Floor-stone",
                       "Floor-tile", "Floor-wood", "Flower", "Fog", "Food-other",
                       "Fruit", "Furniture-other", "Grass", "Gravel", "Ground-other",
                       "Hill", "House", "Leaves", "Light", "Mat",
                       "Metal", "Mirror-stuff", "Moss", "Mountain", "Mud",
                       "Napkin", "Net", "Paper", "Pavement", "Pillow",
                       "Plant-other", "Plastic", "Platform", "Playingfield", "Railing",
                       "Railroad", "River", "Road", "Rock", "Roof",
                       "Rug", "Salad", "Sand", "Sea", "Shelf",
                       "Sky-other", "Skyscraper", "Snow", "Solid-other", "Stairs",
                       "Stone", "Straw", "Structural-other", "Table", "Tent",
                       "Textile-other", "Towel", "Tree", "Vegetable", "Wall-brick",
                       "Wall-concrete", "Wall-other", "Wall-panel", "Wall-stone", "Wall-tile",
                       "Wall-wood", "Water-other", "Waterdrops", "Window-blind", "Window-other",
                       "Wood", "Void"]

        return class_names


    def get_color_map(self):
        color_map = [[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192], [0, 64, 64],
                     [0, 192, 224], [0, 192, 192], [128, 192, 64], [0, 192, 96], [128, 192, 64],
                     [128, 32, 192], [0, 0, 224], [0, 0, 64], [0, 160, 192], [128, 0, 96],
                     [128, 0, 192], [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
                     [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128], [64, 128, 32],
                     [0, 160, 0], [0, 0, 0], [192, 128, 160], [0, 32, 0], [0, 128, 128],
                     [64, 128, 160], [128, 160, 0], [0, 128, 0], [192, 128, 32], [128, 96, 128],
                     [0, 0, 128], [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
                     [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128], [128, 128, 64],
                     [192, 0, 32], [128, 96, 0], [128, 0, 192], [0, 128, 32], [64, 224, 0],
                     [0, 0, 64], [128, 128, 160], [64, 96, 0], [0, 128, 192], [0, 128, 160],
                     [192, 224, 0], [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
                     [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160], [64, 32, 128],
                     [128, 192, 192], [0, 0, 160], [192, 160, 128], [128, 192, 0], [128, 0, 96],
                     [192, 32, 0], [128, 64, 128], [64, 128, 96], [64, 160, 0], [0, 64, 0],
                     [192, 128, 224], [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
                     [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128], [64, 0, 96],
                     [64, 224, 128], [128, 64, 0], [192, 0, 224], [64, 96, 128], [128, 192, 128],
                     [64, 0, 224], [192, 224, 128], [128, 192, 64], [192, 0, 96], [192, 96, 0],
                     [128, 64, 192], [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
                     [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0], [64, 192, 64],
                     [128, 128, 96], [128, 32, 128], [64, 0, 192], [0, 64, 96], [0, 160, 128],
                     [192, 0, 64], [128, 64, 224], [0, 32, 128], [192, 128, 192], [0, 64, 224],
                     [128, 160, 128], [192, 128, 0], [128, 64, 32], [128, 32, 64], [192, 0, 128],
                     [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160], [0, 32, 64],
                     [64, 128, 128], [64, 192, 160], [128, 160, 64], [64, 128, 0], [192, 192, 32],
                     [128, 96, 192], [64, 0, 128], [64, 64, 32], [0, 224, 192], [192, 0, 0],
                     [192, 64, 160], [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
                     [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192], [0, 192, 32],
                     [64, 224, 64], [64, 0, 64], [128, 192, 160], [64, 96, 64], [64, 128, 192],
                     [0, 192, 160], [192, 224, 64], [64, 128, 64], [128, 192, 32], [192, 32, 192],
                     [64, 64, 192], [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
                     [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192], [192, 192, 0],
                     [128, 64, 96], [192, 32, 64], [192, 64, 128], [64, 192, 96], [64, 160, 64],
                     [64, 64, 0], [255, 255, 255]]

        return  color_map
