import os
import argparse
from glob import glob
from multiprocessing.dummy import Pool

import cv2
import numpy as np
import nibabel as nib


parser = argparse.ArgumentParser(description='Slices')
parser.add_argument('--input', type=str, default='/Users/whoami/datasets')
parser.add_argument('--output', type=str, default='/Users/whoami/datasets')
parser.add_argument('--dataset', type=str, default='kits', choices=['lits', 'kits'])
parser.add_argument('--process_num', type=int, default=8)
args = parser.parse_args()


def main():
    os.makedirs(args.output, exist_ok=True)

    if args.dataset == 'lits':
        paths = glob(os.path.join(args.input, "volume-*.nii"))    
    elif args.dataset == 'kits':
        paths = glob(os.path.join(args.input, "case_*/imaging*.nii.gz"))

    pool = Pool(args.process_num)
    pool.map(make_slice, paths)


def make_slice(path):
    if args.dataset == 'lits':
        case, vol, seg = read_lits(path)
    elif args.dataset == 'kits':
        case, vol, seg = read_kits(path)

    for i in range(vol.shape[0]):
        ct_slice = vol[i, ...]
        mask_slice = seg[i, ...]
        if np.any(mask_slice > 1):
            print(f'found slices of {case}-{i}')
            if ct_slice.shape != [512, 512]:
                ct_slice = cv2.resize(ct_slice, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
                mask_slice = cv2.resize(mask_slice, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
            np.savez_compressed(f"{args.output}/{case}_{i}.npz", ct=ct_slice, mask=mask_slice)
    
    print(f'complete making slices of {case}')


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
    main()