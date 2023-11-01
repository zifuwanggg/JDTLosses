import time
import logging

import torch
import torch.nn.functional as F

from utils import get_remaining_time


def train_soft_label(model,
                     data_loader,
                     optimizer,
                     scheduler,
                     device,
                     epoch,
                     args):
    torch.cuda.reset_peak_memory_stats(device)

    model.train()

    end = time.time()
    for iter, (image, label, _, _, _) in enumerate(data_loader):
        start = time.time()
        toprint = f"Epoch: [{epoch}|{args.schedule_config['train_epochs']}], "
        toprint += f"Iter: [{args.schedule_config['curr_iter']}|{args.schedule_config['train_iters']}], "
        toprint += f"Data Time: {(start - end):.6f}, "

        image = image.to(device)
        label = label.to(device)
        keep_mask = label != args.data_config["num_classes"]
        label = get_soft_label(label, keep_mask, args)

        optimizer.zero_grad()
        loss_ce, loss_jdt = model.forward_loss_soft_label(image,
                                                          label,
                                                          keep_mask)
        loss = loss_ce + loss_jdt
        loss.backward()
        optimizer.step()
        scheduler.step()

        end = time.time()
        if iter == 0 or \
           (args.schedule_config["curr_iter"] % args.schedule_config["log_iters"] == 0):
            remaining_time = get_remaining_time(iter,
                                                epoch,
                                                len(data_loader),
                                                end,
                                                args)
            mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            lr = optimizer.param_groups[0]["lr"]

            toprint += f"Batch Time: {(end - start):.6f}, "
            toprint += f"Remaining Time: {remaining_time}, "
            toprint += f"Memory: {mem:.2f}, "
            toprint += f"Learning Rate: {lr:.6f}, "
            toprint += f"Loss({loss.item():.6f}) = "
            toprint += f"CE({loss_ce.item():.6f}) + JDT({loss_jdt.item():.6f})"
            logging.info(toprint)

        args.schedule_config["curr_iter"] += 1
        if args.schedule_config["curr_iter"] > args.schedule_config["train_iters"]:
            break


def get_soft_label(label, keep_mask, args):
    if args.label_config.get("LS", False):
        label_boundary = get_label_boundary(label,
                                            args.data_config["num_classes"],
                                            args.label_config["k"])
        label = smoothing_label(label,
                                label_boundary,
                                args.data_config["num_classes"],
                                args.label_config["epsilon"])
    elif args.label_config.get("MR", False):
        label_background = 1 - label
        label_background[~keep_mask] = args.data_config["num_classes"]
        label = torch.cat((label_background.unsqueeze(1), label.unsqueeze(1)), dim=1)
    else:
        raise NotImplementedError

    return label


def get_label_boundary(label, num_classes, k):
    label_one_hot = \
        F.one_hot(label, num_classes + 1).permute(0, 3, 1, 2).to(torch.float)
    label_one_hot, label_ignore_one_hot = \
        label_one_hot[:, :-1, :, :], label_one_hot[:, -1, :, :]

    label_pool = \
        (-F.max_pool2d(-label_one_hot, kernel_size=k, stride=1, padding=k//2)).to(torch.bool)
    label_ignore_pool = \
        F.max_pool2d(label_ignore_one_hot, kernel_size=k, stride=1, padding=k//2).to(torch.bool)

    label_boundary = torch.any(label_one_hot != label_pool, dim=1)
    label_boundary[label_ignore_pool] = 0

    return label_boundary


def smoothing_label(label,
                    label_boundary,
                    num_classes,
                    epsilon):
    batch_size, crop_h, crop_w = label.shape
    label_boundary = \
        label_boundary.unsqueeze(1).expand(batch_size, num_classes + 1, crop_h, crop_w)

    soft_label = F.one_hot(label, num_classes + 1).permute(0, 3, 1, 2).to(torch.float)
    soft_label[label_boundary] *= (1 - epsilon)
    soft_label[label_boundary] += (epsilon / num_classes)
    soft_label = soft_label[:, :-1, :, :]

    return soft_label
