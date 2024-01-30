from glob import glob

from .base_dataset import BaseDataset


class ADE20K(BaseDataset):
    def __init__(self,
                 train=True,
                 data_dir="/Users/whoami/datasets",
                 in_channels=3,
                 num_classes=150,
                 scale_min=0.5,
                 scale_max=2.0,
                 crop_size=[512, 512],
                 ignore_index=255,
                 reduce_zero_label=True,
                 image_prefix="/images/",
                 image_suffix=".jpg",
                 label_prefix="/annotations/",
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
        train_image_list = glob(f"{self.data_dir}/ade/ADEChallengeData2016/images/training/*.jpg")
        val_image_list = glob(f"{self.data_dir}/ade/ADEChallengeData2016/images/validation/*.jpg")

        assert len(train_image_list) == 20210
        assert len(val_image_list) == 2000

        if self.train:
            return train_image_list
        else:
            return val_image_list


    def get_class_names(self):
        class_names = ["Wall", "Building", "Sky", "Floor", "Tree",
                       "Ceiling", "Road", "Bed ", "Windowpane", "Grass",
                       "Cabinet", "Sidewalk", "Person", "Earth", "Door",
                       "Table", "Mountain", "Plant", "Curtain", "Chair",
                       "Car", "Water", "Painting", "Sofa", "Shelf",
                       "House", "Sea", "Mirror", "Rug", "Field",
                       "Armchair", "Seat", "Fence", "Desk", "Rock",
                       "Wardrobe", "Lamp", "Bathtub", "Railing", "Cushion",
                       "Base", "Box", "Column", "Signboard", "Chest of Drawers",
                       "Counter", "Sand", "Sink", "Skyscraper", "Fireplace",
                       "Refrigerator", "Grandstand", "Path", "Stairs", "Runway",
                       "Case", "Pool Table", "Pillow", "Screen Door", "Stairway",
                       "River", "Bridge", "Bookcase", "Blind", "Coffee Table",
                       "Toilet", "Flower", "Book", "Hill", "Bench",
                       "Countertop", "Stove", "Palm", "Kitchen Island", "Computer",
                       "Swivel Chair", "Boat", "Bar", "Arcade Machine", "Hovel",
                       "Bus", "Towel", "Light", "Truck", "Tower",
                       "Chandelier", "Awning", "Streetlight", "Booth", "Television Receiver",
                       "Airplane", "Dirt Track", "Apparel", "Pole", "Land",
                       "Bannister", "Escalator", "Ottoman", "Bottle", "Buffet",
                       "Poster", "Stage", "Van", "Ship", "Fountain",
                       "Conveyer Belt", "Canopy", "Washer", "Plaything", "Swimming Pool",
                       "Stool", "Barrel", "Basket", "Waterfall", "Tent",
                       "Bag", "Minibike", "Cradle", "Oven", "Ball",
                       "Food", "Step", "Tank", "Trade Name", "Microwave",
                       "Pot", "Animal", "Bicycle", "Lake", "Dishwasher",
                       "Screen", "Blanket", "Sculpture", "Hood", "Sconce",
                       "Vase", "Traffic Light", "Tray", "Ashcan", "Fan",
                       "Pier", "Crt Screen", "Plate", "Monitor", "Bulletin Board",
                       "Shower", "Radiator", "Glass", "Clock", "Flag",
                       "Void"]

        return class_names


    def get_color_map(self):
        color_map = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
                     [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7],
                     [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51],
                     [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                     [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71],
                     [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255],
                     [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10],
                     [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                     [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255],
                     [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15],
                     [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0],
                     [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                     [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112],
                     [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0],
                     [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173],
                     [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                     [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184],
                     [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204], [0, 255, 194],
                     [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255],
                     [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                     [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0], [8, 184, 170],
                     [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255],
                     [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255],
                     [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                     [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0], [255, 0, 235],
                     [245, 0, 255], [255, 0, 122], [255, 245, 0], [10, 190, 212], [214, 255, 0],
                     [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255], [0, 41, 255],
                     [0, 255, 204], [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                     [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255], [184, 255, 0],
                     [0, 133, 255], [255, 214, 0], [25, 194, 194], [102, 255, 0], [92, 0, 255],
                     [255, 255, 255]]

        return color_map
