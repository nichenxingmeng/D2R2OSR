"""Generate a BasicSR meta_info file (path + shape) for a folder of images.

Usage:
    python scripts/generate_meta_info.py --gt-folder GT_FOLDER --meta-info-txt OUT.txt
"""
import argparse
from os import path as osp

from PIL import Image

from basicsr.utils import scandir


def generate_meta_info(gt_folder, meta_info_txt):
    img_list = sorted(list(scandir(gt_folder)))

    with open(meta_info_txt, 'w') as f:
        for idx, img_path in enumerate(img_list):
            img = Image.open(osp.join(gt_folder, img_path))  # lazy load
            width, height = img.size
            mode = img.mode
            if mode == 'RGB':
                n_channel = 3
            elif mode == 'L':
                n_channel = 1
            else:
                raise ValueError(f'Unsupported mode {mode}.')

            info = f'{img_path} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt-folder', required=True,
                        help='folder of GT images to scan')
    parser.add_argument('--meta-info-txt', required=True,
                        help='output meta_info .txt path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_meta_info(args.gt_folder, args.meta_info_txt)
