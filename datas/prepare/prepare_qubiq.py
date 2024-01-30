import os
import sys
from glob import glob

import cv2
import sitk
import numpy as np


def prepare_qubiq(data_dir, dataset, task):
    def compute_dice(a, b):
        cardinality = np.sum(a + b)
        difference = np.sum(np.abs(a - b))
        if cardinality <= 0:
            return 1

        score = (cardinality - difference) / (cardinality)

        if score <= 0:
            return 1

        return score

    num_cases = {"brain-growth": 39, "brain-tumor": 32, "kidney": 24, "prostate": 55}
    num_raters = {"brain-growth": 7, "brain-tumor": 3, "kidney": 3, "prostate": 6}

    data_dir = os.path.join(data_dir, f"qubiq-raw/{dataset}")

    dice_all_cases = []

    for case in range(1, num_cases[dataset] + 1):
        case_dir = os.path.join(data_dir, f"case{str(case).zfill(2)}")
        src_image_path = os.path.join(case_dir, "image.nii.gz")
        src_label_path = glob(f"{case_dir}/task{str(task).zfill(2)}*")

        if dataset == "prostate" and case in [9]:
            continue

        assert len(src_label_path) == num_raters[dataset], f"{case_dir}"

        dst_case_dir = case_dir.replace("qubiq-raw", "qubiq")
        if case == 55 and dataset == "prostate":
            dst_case_dir = dst_case_dir.replace("55", "09")
        dst_image_path = os.path.join(dst_case_dir, "image.npz")
        dst_label_path = os.path.join(dst_case_dir, f"task{task}_label.npz")
        os.makedirs(dst_case_dir, exist_ok=True)

        image_sitk = sitk.ReadImage(src_image_path)
        image_np = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
        if len(image_np.shape) == 3:
            if image_np.shape[0] == 1:
                image_np = image_np[0, :, :]
            else:
                image_np = np.moveaxis(image_np, 0, -1)

        image_scaled = 255 * (image_np / image_np.max())
        image_resized = cv2.resize(image_scaled, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)

        label = np.zeros((256, 256), np.uint8)
        for i, label_path in enumerate(src_label_path):
            label_sitk = sitk.ReadImage(label_path)
            label_np = sitk.GetArrayFromImage(label_sitk).astype(np.uint8)
            if len(label_np.shape) == 3:
                label_np = label_np[0, :, :]

            label_resized = cv2.resize(label_np, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
            label += label_resized

            dst_label_i_path = dst_label_path.replace(f"task{task}_label.npz", f"task{task}_seg{i}_label.npz")
            np.savez_compressed(dst_label_i_path, data=label_resized)

        label_majority = (label >= (num_raters[dataset] // 2 + 1)).astype(np.float32)

        dice_total = 0
        label_weighted = np.zeros((256, 256), np.float32)
        for i, label_path in enumerate(src_label_path):
            label_sitk = sitk.ReadImage(label_path)
            label_np = sitk.GetArrayFromImage(label_sitk).astype(np.uint8)
            if len(label_np.shape) == 3:
                label_np = label_np[0, :, :]

            label_resized = cv2.resize(label_np, dsize=(256, 256), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            dice = compute_dice(label_resized, label_majority)
            dice_total += dice
            label_weighted += dice * label_resized

        dice_all_cases.append(dice_total / num_raters[dataset])
        label_majority = label_majority.astype(np.uint8)
        label_weighted /= dice_total

        dst_label_majority_path = dst_label_path.replace(f"task{task}_label.npz", f"task{task}_majority_label.npz")
        dst_label_weighted_path = dst_label_path.replace(f"task{task}_label.npz", f"task{task}_weighted_label.npz")

        np.savez_compressed(dst_image_path, data=image_resized)
        np.savez_compressed(dst_label_path, data=label)
        np.savez_compressed(dst_label_majority_path, data=label_majority)
        np.savez_compressed(dst_label_weighted_path, data=label_weighted)

    sum(dice_all_cases) / len(dice_all_cases)


if __name__ == '__main__':
    prepare_qubiq(sys.argv[1], sys.argv[2], sys.argv[3])
