import os
import time
import yaml
import logging
import argparse
from glob import glob

import torch
import torch.nn as nn
from torch import optim
import torch.distributed as dist
from timm.layers.norm_act import convert_sync_batchnorm

from models.segmentor import Segmentor
from datas.dataloader import get_dataloader
from losses.get_criterion import get_criterion
from train_hard_label import train_hard_label
from train_soft_label import train_soft_label
from train_kd import train_kd
from test import test, test_medical
from utils import synchronize, add_handler, load_checkpoint

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def get_args():
    parser = argparse.ArgumentParser(description="Main", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_checkpoint", type=str, default="")
    parser.add_argument("--teacher_checkpoint", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="/Users/whoami/datasets")
    parser.add_argument("--output_dir", type=str, default="/Users/whoami/outputs")
    parser.add_argument("--model_yaml", type=str, default="deeplabv3plus_resnet101d")
    parser.add_argument("--teacher_model_yaml", type=str, default="deeplabv3_resnet50d")
    parser.add_argument("--data_yaml", type=str, default="cityscapes")
    parser.add_argument("--label_yaml", type=str, default="hard")
    parser.add_argument("--loss_yaml", type=str, default="jaccard_ic_present_all")
    parser.add_argument("--schedule_yaml", type=str, default="40k_iters")
    parser.add_argument("--optim_yaml", type=str, default="adamw_lr6e-5")
    parser.add_argument("--test_yaml", type=str, default="test_iou")

    args = parser.parse_args()

    args.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    args.distributed = args.world_size > 1
    args.main_process = args.local_rank == 0

    with open(f"configs/models/{args.model_yaml}.yaml", mode="r") as model_yaml, \
         open(f"configs/models/{args.teacher_model_yaml}.yaml", mode="r") as teacher_model_yaml, \
         open(f"configs/datas/{args.data_yaml}.yaml", mode="r") as data_yaml, \
         open(f"configs/labels/{args.label_yaml}.yaml", mode="r") as label_yaml, \
         open(f"configs/losses/{args.loss_yaml}.yaml", mode="r") as loss_yaml, \
         open(f"configs/schedules/{args.schedule_yaml}.yaml", mode="r") as schedule_yaml, \
         open(f"configs/optims/{args.optim_yaml}.yaml", mode="r") as optim_yaml, \
         open(f"configs/test/{args.test_yaml}.yaml", mode="r") as test_yaml:
        args.model_config = yaml.load(model_yaml, Loader=yaml.Loader)
        args.teacher_model_config = yaml.load(teacher_model_yaml, Loader=yaml.Loader)
        args.data_config = yaml.load(data_yaml, Loader=yaml.Loader)
        args.label_config = yaml.load(label_yaml, Loader=yaml.Loader)
        args.loss_config = yaml.load(loss_yaml, Loader=yaml.Loader)
        args.schedule_config = yaml.load(schedule_yaml, Loader=yaml.Loader)
        args.optim_config = yaml.load(optim_yaml, Loader=yaml.Loader)
        args.test_config = yaml.load(test_yaml, Loader=yaml.Loader)

    return args


def main(args):
    device = torch.device("cuda:{}".format(args.local_rank))
    torch.backends.cudnn.benchmark = True

    criterion_ce, criterion_kl, criterion_jdt = get_criterion(args)

    model = Segmentor(args.model_config, args.data_config, args.label_config, args.loss_config, criterion_ce, criterion_kl, criterion_jdt).to(device)
    params_list = model.get_params_list(args.optim_config["lr"], args.optim_config["multiplier"])

    if args.label_config.get("KD", False):
        teacher = Segmentor(args.teacher_model_config, args.data_config, args.label_config, args.loss_config, None, None, None).to(device)
        _ = load_checkpoint(args.teacher_checkpoint, teacher, None, None, device)
        logging.info(f"Teacher checkpoint at {args.teacher_checkpoint} is loaded")

    args.schedule_config["batch_size"] //= args.world_size
    train_loader, test_loader = get_dataloader(args)

    if args.optim_config["optimizer"] == "sgd":
        optimizer = optim.SGD(params_list, lr=args.optim_config["lr"], weight_decay=args.optim_config["weight_decay"], momentum=args.optim_config["momentum"])
    elif args.optim_config["optimizer"] == "adamw":
        optimizer = optim.AdamW(params_list, lr=args.optim_config["lr"], weight_decay=args.optim_config["weight_decay"])
    else:
        raise NotImplementedError

    if args.schedule_config["by_epoch"]:
        args.schedule_config["train_iters"] = args.schedule_config["train_epochs"] * len(train_loader)
    else:
        args.schedule_config["train_epochs"] = args.schedule_config["train_iters"] // len(train_loader) + 1

    warmup_iters = args.schedule_config["warmup_ratio"] * args.schedule_config["train_iters"]

    lr_lambda = lambda iteration: max(args.optim_config["start_lr"] / args.optim_config["lr"], iteration / warmup_iters) \
        if iteration < warmup_iters else (1. - (iteration - warmup_iters) / (args.schedule_config["train_iters"] - warmup_iters)) ** args.optim_config["power"]

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if os.path.isfile(args.model_checkpoint) and args.model_checkpoint[-4:] == ".pth":
        ckp_path = args.model_checkpoint
    else:
        avail = glob(os.path.join(args.output_dir, 'checkpoint*.pth'))
        avail = [(int(f[-len('.pth') - 3 : -len('.pth')]), f) for f in avail]
        avail = sorted(avail)
        ckp_path = avail[-1][1] if avail else ""

    if os.path.isfile(ckp_path):
        args.schedule_config["start_epoch"] = load_checkpoint(ckp_path, model, optimizer, scheduler, device) + 1
        logging.info(f"Checkpoint at {ckp_path} is loaded")
    else:
        args.schedule_config["start_epoch"] = 1
        logging.info(f"No checkpoint is found")

    args.schedule_config["curr_iter"] = 1 + len(train_loader) * (args.schedule_config["start_epoch"] - 1)

    # Do not use `nn.SyncBatchNorm.convert_sync_batchnorm`
    # https://github.com/rwightman/pytorch-image-models/issues/1254
    if args.distributed:
        if args.label_config.get("KD", False):
            teacher = convert_sync_batchnorm(teacher)
            teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.local_rank])
        model = convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    args.schedule_config["start_time"] = time.time()
    toprint = "\n"
    toprint += "".join(f"{arg}: {value}\n" for arg, value in vars(args).items())
    logging.info(toprint)

    for epoch in range(args.schedule_config["start_epoch"], args.schedule_config["train_epochs"] + 1):
        logging.info(f"Begin train epoch {epoch}")

        if args.distributed:
            train_loader.sampler.set_epoch(epoch - 1)

        if args.label_config.get("HARD", False):
            train_hard_label(model, train_loader, optimizer, scheduler, device, epoch, args)
        elif args.label_config.get("LS", False) or args.label_config.get("MR", False):
            train_soft_label(model, train_loader, optimizer, scheduler, device, epoch, args)
        elif args.label_config.get("KD", False):
            train_kd(model, teacher, train_loader, optimizer, scheduler, device, epoch, args)
        else:
            raise NotImplementedError

        if args.main_process and epoch % args.schedule_config["save_epochs"] == 0:
            state = {"epoch": epoch,
                     "state_dict": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": scheduler.state_dict()}

            ckp_name = f"checkpoint{epoch:03d}.pth"
            old_ckp_name = f"checkpoint{(epoch - 1):03d}.pth"
            ckp_path = os.path.join(args.output_dir, ckp_name)
            old_ckp_path = os.path.join(args.output_dir, old_ckp_name)

            torch.save(state, ckp_path)

            if args.test_config["ITER"]["remove_old"] and os.path.isfile(ckp_path) and os.path.isfile(old_ckp_path):
                os.remove(old_ckp_path)

        if args.main_process:
            if epoch == args.schedule_config["train_epochs"] or (args.test_config["ITER"]["test_epochs"] > 0 and epoch % args.test_config["ITER"]["test_epochs"] == 0):
                logging.info(f"Begin test epoch {epoch}")

                with torch.no_grad():
                    if args.data_config["dataset"] in ["lits", "kits"] or "qubiq" in args.data_config["dataset"]:
                        test_medical(model, test_loader, device, epoch, args)
                    else:
                        test(model, test_loader, device, epoch, args)

        logging.info(f"Finish epoch {epoch}\n")


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.cuda.device(args.local_rank)
    if args.distributed:
        dist.init_process_group(backend="nccl", init_method="env://", world_size=args.world_size, rank=args.local_rank)
        synchronize()

    if args.local_rank == 0:
        add_handler(args.output_dir, "train.log", mode="a")
    logging.getLogger().setLevel(logging.DEBUG)

    main(args)
