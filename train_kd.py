import time
import logging

import torch

from kde import kde
from utils import get_remaining_time
from train_soft_label import get_label_boundary


def train_kd(model,
             teacher,
             data_loader,
             optimizer,
             scheduler,
             device,
             epoch,
             args):
    torch.cuda.reset_peak_memory_stats(device)

    model.train()
    teacher.eval()

    end = time.time()
    for iter, (image, label, _, _, _) in enumerate(data_loader):
        start = time.time()
        toprint = f"Epoch: [{epoch}|{args.schedule_config['train_epochs']}], "
        toprint += f"Iter: [{args.schedule_config['curr_iter']}|{args.schedule_config['train_iters']}], "
        toprint += f"Data Time: {(start - end):.6f}, "

        image = image.to(device)
        label = label.to(device)
        keep_mask = label != args.data_config["num_classes"]
        prob_teacher, aux_prob_teacher = get_soft_label(teacher,
                                                        image,
                                                        label,
                                                        args)

        optimizer.zero_grad()
        loss_ce, loss_jdt = model.forward_loss_kd(image,
                                                  label,
                                                  prob_teacher,
                                                  aux_prob_teacher,
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


def get_soft_label(teacher,
                   image,
                   label,
                   args):
    with torch.no_grad():
        logits_teacher, aux_logits_teacher = teacher(image)
        prob_teacher = \
            (logits_teacher / args.label_config["T"]).log_softmax(dim=1).exp()
        aux_prob_teacher = \
            (aux_logits_teacher / args.label_config["T"]).log_softmax(dim=1).exp()

        if args.label_config["kde"]:
            batch_size, crop_h, crop_w = label.shape
            label_boundary = get_label_boundary(label,
                                                args.data_config["num_classes"],
                                                args.label_config["k"])
            label_boundary = \
                label_boundary.unsqueeze(1).expand(batch_size,
                                                   args.data_config["num_classes"],
                                                   crop_h,
                                                   crop_w)
            prob_teacher, aux_prob_teacher = kde(prob_teacher,
                                                 aux_prob_teacher,
                                                 label,
                                                 label_boundary,
                                                 args)

    return prob_teacher, aux_prob_teacher
