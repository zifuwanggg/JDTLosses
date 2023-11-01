import os
import sys
import logging
import datetime
from collections import OrderedDict

import cv2

import torch
import torch.distributed as dist

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def synchronize():
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    if dist.get_world_size() == 1:
        return

    dist.barrier()


def add_handler(output_dir, log_name, mode="a"):
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

    file_handler = logging.FileHandler(os.path.join(output_dir, log_name), mode=mode)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(stream_handler)


def load_checkpoint(ckp_path,
                    model,
                    optimizer=None,
                    scheduler=None,
                    device="cuda"):
    checkpoint = torch.load(ckp_path, map_location=device)
    start_epoch = checkpoint["epoch"]
    state_dict = checkpoint["state_dict"]

    try:
        model.load_state_dict(state_dict)
    except:
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler:
        scheduler.load_state_dict(checkpoint["lr_scheduler"])

    return start_epoch


def get_remaining_time(iter,
                       epoch,
                       epoch_iters,
                       end,
                       args):
    passed_iter = 1 + iter + epoch_iters * \
        (epoch - args.schedule_config["start_epoch"])
    remaining_iter = args.schedule_config["train_iters"] - \
        args.schedule_config["curr_iter"]
    seconds = remaining_iter * \
        ((end - args.schedule_config["start_time"]) / passed_iter)
    remaining_time = str(datetime.timedelta(seconds=int(seconds)))

    return remaining_time
