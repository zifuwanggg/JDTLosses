import os
import sys
from glob import glob
from multiprocessing.dummy import Pool

import cv2
import numpy as np
import nibabel as nib


def prepare_lits_kits(data_dir, dataset):
    os.makedirs(f"{data_dir}/{dataset}/train", exist_ok=True)

    if dataset == 'lits':
        src_files = glob("{data_dir}/{dataset}/volume-*.nii")
    elif dataset == 'kits':
        src_files = glob("{data_dir}/{dataset}/case_*/imaging*.nii.gz")

    pool = Pool(8)
    pool.map(make_slice, src_files)


def make_slice(src_file, data_dir, dataset):
    if dataset == 'lits':
        case, vol, seg = read_lits(src_file)
    elif dataset == 'kits':
        case, vol, seg = read_kits(src_file)

    for i in range(vol.shape[0]):
        ct_slice = vol[i, ...]
        mask_slice = seg[i, ...]
        if np.any(mask_slice > 1):
            if ct_slice.shape != [512, 512]:
                ct_slice = cv2.resize(ct_slice, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
                mask_slice = cv2.resize(mask_slice, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
            np.savez_compressed(f"{data_dir}/{dataset}/train/{case}_{i}.npz", ct=ct_slice, mask=mask_slice)


def read_lits(path):
    case = path.split('-')[-1].split('.')[0]
    vol = nib.load(path).get_fdata()
    seg = nib.load(path.replace('volume', 'segmentation')).get_fdata().astype('int8')
    vol = np.transpose(vol, (2, 0, 1))
    seg = np.transpose(seg, (2, 0, 1))
    return case, vol, seg


def read_kits(path):
    dir = os.path.dirname(path)
    case = os.path.split(dir)[-1][-5:]
    vol = nib.load(path).get_fdata()
    seg = nib.load(os.path.join(dir, 'segmentation.nii.gz')).get_fdata().astype('int8')
    return case, vol, seg


if __name__ == '__main__':
    prepare_lits_kits(sys.argv[1], sys.argv[2])
