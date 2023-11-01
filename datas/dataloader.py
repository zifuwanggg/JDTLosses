from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .datasets.cityscapes import Cityscapes
from .datasets.nighttime_driving import NighttimeDriving
from .datasets.dark_zurich import DarkZurich
from .datasets.mapillary import Mapillary
from .datasets.camvid import CamVid
from .datasets.ade20k import ADE20K
from .datasets.coco_stuff import COCOStuff
from .datasets.voc import VOC
from .datasets.context import Context
from .datasets.land import Land
from .datasets.road import Road
from .datasets.building import Building
from .datasets.lits import LiTS
from .datasets.kits import KiTS
from .datasets.qubiq_braingrowth import QUBIQBrainGrowth
from .datasets.qubiq_braintumor import QUBIQBrainTumor
from .datasets.qubiq_kidney import QUBIQKidney
from .datasets.qubiq_prostate import QUBIQProstate


def get_dataloader(args):
    if args.data_config["dataset"] not in \
        ["cityscapes", "nighttime", "darkzurich", "mapillary", "camvid",
         "ade", "coco", "voc", "context",
         "land", "road", "building",
         "lits", "kits",
         "qubiq/brain-growth", "qubiq/brain-tumor", "qubiq/kidney", "qubiq/prostate"]:
        raise NotImplementedError

    train_dataset = get_dataset(True, args.data_dir, args.data_config)
    test_dataset = get_dataset(False, args.data_dir, args.data_config)

    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_dataset,
                                           shuffle=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.schedule_config["batch_size"],
                                  sampler=train_sampler,
                                  num_workers=args.schedule_config["num_workers"],
                                  pin_memory=True,
                                  drop_last=True)
    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.schedule_config["batch_size"],
                                  shuffle=True,
                                  num_workers=args.schedule_config["num_workers"],
                                  pin_memory=True,
                                  drop_last=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.schedule_config["num_workers"],
                             pin_memory=True,
                             drop_last=False)

    return train_loader, test_loader


dataset_objects = {"cityscapes": Cityscapes,
                   "nighttime": NighttimeDriving,
                   "darkzurich": DarkZurich,
                   "mapillary": Mapillary,
                   "camvid": CamVid,
                   "ade": ADE20K,
                   "coco": COCOStuff,
                   "voc": VOC,
                   "context": Context,
                   "land": Land,
                   "road": Road,
                   "building": Building,
                   "lits": LiTS,
                   "kits": KiTS,
                   "qubiq/brain-growth": QUBIQBrainGrowth,
                   "qubiq/brain-tumor": QUBIQBrainTumor,
                   "qubiq/kidney": QUBIQKidney,
                   "qubiq/prostate": QUBIQProstate}


def get_dataset(train, data_dir, data_config):
    if data_config["dataset"] in ["cityscapes", "nighttime", "darkzurich", "mapillary", "camvid",
                                  "ade", "coco", "voc", "context"]:
        return dataset_objects[data_config["dataset"]] (
            train=train,
            data_dir=data_dir,
            in_channels=data_config["in_channels"],
            num_classes=data_config["num_classes"],
            scale_min=data_config["scale_min"],
            scale_max=data_config["scale_max"],
            crop_size=data_config["crop_size"],
            ignore_index=data_config["ignore_index"],
            reduce_zero_label=data_config["reduce_zero_label"],
            reduce_panoptic_zero_label=data_config["reduce_panoptic_zero_label"],
            image_prefix=data_config["image_prefix"],
            image_suffix=data_config["image_suffix"],
            label_prefix=data_config["label_prefix"],
            label_suffix=data_config["label_suffix"])
    elif data_config["dataset"] in ["land", "road", "building"]:
        return dataset_objects[data_config["dataset"]](
            train=train,
            data_dir=data_dir,
            fold=data_config["fold"],
            in_channels=data_config["in_channels"],
            num_classes=data_config["num_classes"],
            scale_min=data_config["scale_min"],
            scale_max=data_config["scale_max"],
            crop_size=data_config["crop_size"],
            ignore_index=data_config["ignore_index"],
            reduce_zero_label=data_config["reduce_zero_label"],
            reduce_panoptic_zero_label=data_config["reduce_panoptic_zero_label"],
            image_prefix=data_config["image_prefix"],
            image_suffix=data_config["image_suffix"],
            label_prefix=data_config["label_prefix"],
            label_suffix=data_config["label_suffix"])
    elif data_config["dataset"] in ["lits", "kits"]:
        return dataset_objects[data_config["dataset"]](
            train=train,
            data_dir=data_dir,
            fold=data_config["fold"],
            num_cases=data_config["num_cases"],
            in_channels=data_config["in_channels"],
            num_classes=data_config["num_classes"],
            scale_min=data_config["scale_min"],
            scale_max=data_config["scale_max"],
            window_low=data_config["window_low"],
            window_high=data_config["window_high"],
            crop_size=data_config["crop_size"],
            ignore_index=data_config["ignore_index"],
            weighted=data_config["weighted"],
            reduce_zero_label=data_config["reduce_zero_label"],
            reduce_panoptic_zero_label=data_config["reduce_panoptic_zero_label"],
            image_prefix=data_config["image_prefix"],
            image_suffix=data_config["image_suffix"],
            label_prefix=data_config["label_prefix"],
            label_suffix=data_config["label_suffix"])
    elif data_config["dataset"] in ["qubiq/brain-growth",
                                    "qubiq/brain-tumor",
                                    "qubiq/kidney",
                                    "qubiq/prostate"]:
        return dataset_objects[data_config["dataset"]](
            train=train,
            data_dir=data_dir,
            fold=data_config["fold"],
            task=data_config["task"],
            num_cases=data_config["num_cases"],
            num_raters=data_config["num_raters"],
            in_channels=data_config["in_channels"],
            num_classes=data_config["num_classes"],
            scale_min=data_config["scale_min"],
            scale_max=data_config["scale_max"],
            crop_size=data_config["crop_size"],
            ignore_index=data_config["ignore_index"],
            weighted=data_config["weighted"],
            reduce_zero_label=data_config["reduce_zero_label"],
            reduce_panoptic_zero_label=data_config["reduce_panoptic_zero_label"],
            image_prefix=data_config["image_prefix"],
            image_suffix=data_config["image_suffix"],
            label_prefix=data_config["label_prefix"],
            label_suffix=data_config["label_suffix"])
