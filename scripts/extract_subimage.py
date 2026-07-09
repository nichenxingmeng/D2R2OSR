"""Multi-thread tool to crop large ERP images into training sub-images.

Usage:
    python scripts/extract_subimage.py --input-folder HR --save-folder HR_sub \
        --crop-size 512 --step 256 --wh 2048 1024
"""
import argparse
import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
from basicsr.utils import scandir


def main():
    args = parse_args()
    opt = {
        'n_thread': args.n_thread,
        'compression_level': args.compression_level,
        'input_folder': args.input_folder,
        'save_folder': args.save_folder,
        'wh': tuple(args.wh) if args.wh else None,
        'scale': args.scale,
        'crop_size': args.crop_size,
        'step': args.step,
        'thresh_size': args.thresh_size,
    }
    extract_subimages(opt)


def extract_subimages(opt):
    """Crop images to subimages.
    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
            n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        sys.exit(1)

    img_list = list(scandir(input_folder, full_path=True))

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker(path, opt):
    """Worker for each process.
    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            step (int): Step for overlapped sliding window.
            thresh_size (int): Threshold size. Patches whose size is lower
                than thresh_size will be dropped.
            save_folder (str): Path to save folder.
            compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.
    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))
    input_shape = opt['wh']
    scale = opt['scale']

    img_name = img_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if input_shape is not None:
        img = cv2.resize(img, input_shape, cv2.INTER_LINEAR)
    h, w = img.shape[0:2]

    # top_margin = 160 # 部分数据集上下模糊进行裁剪，默认为0
    # bottom_margin = 160
    top_margin = 0 # 部分数据集上下模糊进行裁剪，默认为0
    bottom_margin = 0
    h_step = crop_size - top_margin - bottom_margin
    h_space = np.arange(
        top_margin,
        h - crop_size - bottom_margin + 1,
        h_step
    )

    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(
                osp.join(opt['save_folder'], f'{img_name}_s{index:03d}_hw~_{int(x*scale)}_{int(y*scale)}_~{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processing {img_name} ...'
    return process_info


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-folder', required=True)
    parser.add_argument('--save-folder', required=True)
    parser.add_argument('--crop-size', type=int, default=512)
    parser.add_argument('--step', type=int, default=256)
    parser.add_argument('--thresh-size', type=int, default=0)
    parser.add_argument('--scale', type=int, default=1,
                        help='scale factor between this folder and the HR '
                        'reference (for naming crop offsets consistently '
                        'across LR/HR pairs)')
    parser.add_argument('--wh', type=int, nargs=2, default=None,
                        help='resize each image to (W, H) before cropping; '
                        'omit to crop at native resolution')
    parser.add_argument('--n-thread', type=int, default=20)
    parser.add_argument('--compression-level', type=int, default=3)
    return parser.parse_args()


if __name__ == '__main__':
    main()