import os
import time
import pickle
import logging

import cv2
import numpy as np

from metrics.metric_group import MetricGroup

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def test(model, data_loader, device, epoch, args):
    model.eval()

    dataset = data_loader.dataset
    metric_group = MetricGroup(accuracyD=args.test_config["METRIC"]["accuracyD"],
                               accuracyI=args.test_config["METRIC"]["accuracyI"],
                               accuracyC=args.test_config["METRIC"]["accuracyC"],
                               ECED=args.test_config["METRIC"]["ECED"],
                               ECEI=args.test_config["METRIC"]["ECEI"],
                               SCED=args.test_config["METRIC"]["SCED"],
                               SCEI=args.test_config["METRIC"]["SCEI"],
                               q=args.test_config["METRIC"]["q"],
                               binary=args.data_config["num_classes"] == 2,
                               num_bins=args.test_config["METRIC"]["num_bins"],
                               num_classes=args.data_config["num_classes"],
                               ignore_index=args.data_config["num_classes"])

    end = time.time()
    for iter, (image, label, image_file) in enumerate(data_loader):
        start = time.time()
        toprint = f"Epoch: [{epoch}|{args.schedule_config['train_epochs']}], "
        toprint += f"Iter: [{iter}|{len(data_loader)}], "
        toprint += f"Data Time: {(start - end):.6f}, "

        prob = model.multi_scale_predict(image,
                                         device,
                                         args.data_config["num_classes"],
                                         args.data_config["crop_size"],
                                         args.test_config["INF"]["flip"],
                                         args.test_config["INF"]["ratios"],
                                         args.test_config["INF"]["stride_rate"])

        mid = time.time()
        toprint += f"Batch Time: {(mid - start):.6f}, "

        metric_group.add(prob, label, image_file)

        if args.test_config["ITER"]["save_pred"]:
            save_pred(prob, label, image_file, dataset.color_map, args)

        end = time.time()
        toprint += f"Metric Time: {(end - mid):.6f}"

        if iter % args.test_config["ITER"]["log_iters"] == 0:
            logging.info(toprint)

    toprint = "\n"
    results = metric_group.value()
    for key, value in results.items():
        toprint += f"{key}: {value:.2f}\n"
    toprint = toprint[:-2]
    logging.info(toprint)

    with open(os.path.join(args.output_dir, "metric_group.pkl"), "wb") as pkl:
        pickle.dump(metric_group, pkl)


def test_medical(model, data_loader, device, epoch, args):
    model.eval()

    val_cases = [False] * (args.data_config["num_cases"] + 1)

    metric_groups = {}
    dataset = data_loader.dataset
    for case in range(args.data_config["num_cases"] + 1):
        metric_groups[case] = MetricGroup(accuracyD=args.test_config["METRIC"]["accuracyD"],
                                          accuracyI=args.test_config["METRIC"]["accuracyI"],
                                          accuracyC=args.test_config["METRIC"]["accuracyC"],
                                          ECED=args.test_config["METRIC"]["ECED"],
                                          ECEI=args.test_config["METRIC"]["ECEI"],
                                          SCED=args.test_config["METRIC"]["SCED"],
                                          SCEI=args.test_config["METRIC"]["SCEI"],
                                          q=args.test_config["METRIC"]["q"],
                                          binary=args.data_config["num_classes"] == 2,
                                          num_bins=args.test_config["METRIC"]["num_bins"],
                                          num_classes=args.data_config["num_classes"],
                                          ignore_index=args.data_config["num_classes"])

    end = time.time()
    for iter, (image, label, image_file) in enumerate(data_loader):
        start = time.time()
        toprint = f"Epoch: [{epoch}|{args.schedule_config['train_epochs']}], "
        toprint += f"Iter: [{iter}|{len(data_loader)}], "
        toprint += f"Data Time: {(start - end):.6f}, "

        prob = model.multi_scale_predict(image,
                                         device,
                                         args.data_config["num_classes"],
                                         args.data_config["crop_size"],
                                         args.test_config["INF"]["flip"],
                                         args.test_config["INF"]["ratios"],
                                         args.test_config["INF"]["stride_rate"])

        mid = time.time()
        toprint += f"Batch Time: {(mid - start):.6f}, "

        if "qubiq" in args.data_config["dataset"]:
            ignore = label == args.data_config["num_classes"]
            label = (label >= (args.data_config["num_raters"] // 2 + 1)).long()
            label[ignore] = args.data_config["num_classes"]
            label = label.long()

        for i in range(image.shape[0]):
            if args.data_config["dataset"] in ["lits", "kits"]:
                case = int(image_file[i].split("/")[-1].split("_")[0])
            elif "qubiq" in args.data_config["dataset"]:
                case = int(image_file[i].split("/")[-2][-2:])
            else:
                raise NotImplementedError

            val_cases[case] = True

            prob_i = prob[i, :, :, :].unsqueeze(0)
            label_i = label[i, :, :].unsqueeze(0)
            metric_groups[case].add(prob_i, label_i, image_file)

        if args.test_config["ITER"]["save_pred"]:
            save_pred(prob, label, image_file, dataset.color_map, args)

        end = time.time()
        toprint += f"Metric Time: {(end - mid):.6f}"

        if iter % args.test_config["log_iters"] == 0:
            logging.info(toprint)

    results = {}
    for i, case in enumerate(val_cases):
        if case:
            results_case = metric_groups[i].value()
            for key, value in results_case.items():
                if key not in results:
                    results[key] = value
                else:
                    results[key] += value

    for key in results:
        results[key] /= sum(val_cases)

    toprint = ""
    for key, value in results.items():
        toprint += f"{key}: {value:.2f}\n"
    toprint = toprint[:-2]
    logging.info(toprint)

    with open(os.path.join(args.output_dir, f"metric_groups.pkl"), "wb") as pkl:
        pickle.dump(metric_groups, pkl)


def save_pred(prob, label, image_file, color_map, args):
    pred_dir = os.path.join(args.output_dir, "pred")
    os.makedirs(pred_dir, exist_ok=True)

    pred = prob.argmax(1)
    pred[label == args.data_config["num_classes"]] = args.data_config["num_classes"]

    for i in range(label.shape[0]):
        pred_i = pred[i, :, :].cpu().numpy()
        label_i = label[i, :, :].cpu().numpy()

        pred_rgb = np.zeros((label.shape[1], label.shape[2], 3))
        label_rgb = np.zeros((label.shape[1], label.shape[2], 3))

        for j in range(len(color_map)):
            pred_rgb[:, :, 2][pred_i == j] = color_map[j][0]
            pred_rgb[:, :, 1][pred_i == j] = color_map[j][1]
            pred_rgb[:, :, 0][pred_i == j] = color_map[j][2]
            label_rgb[:, :, 2][label_i == j] = color_map[j][0]
            label_rgb[:, :, 1][label_i == j] = color_map[j][1]
            label_rgb[:, :, 0][label_i == j] = color_map[j][2]

        label_file = os.path.join(pred_dir, image_file[i].split("/")[-1])
        label_file = label_file.replace(".jpg", ".png")
        pred_file = label_file.replace(".png", "_pred.png")

        cv2.imwrite(pred_file, pred_rgb)
        cv2.imwrite(label_file, label_rgb)
