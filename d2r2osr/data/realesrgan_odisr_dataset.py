import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
from basicsr.data.data_util import scandir
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from .utils import paired_random_crop

@DATASET_REGISTRY.register()
class RealESRGANODISRDataset(data.Dataset):
    """Dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(RealESRGANODISRDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']

        if opt['mode'] != 'train':
            self.lq_folder = opt['dataroot_lq']
            # file client (lmdb io backend)
            if self.io_backend_opt['type'] == 'lmdb':
                self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
                self.io_backend_opt['client_keys'] = ['gt', 'lq']
                self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
            elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
                self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                              self.opt['meta_info_file'], self.filename_tmpl)
            else:
                gt_paths = sorted(list(scandir(self.gt_folder, full_path=True)))
                lq_paths = sorted(list(scandir(self.lq_folder, full_path=True)))
                
                assert len(gt_paths) == len(lq_paths), \
                    f"GT and LQ datasets have different lengths"
                
                self.paths = []
                
                for gt_path in gt_paths:
                    filename = os.path.basename(gt_path)
                    lq_path = os.path.join(self.lq_folder, filename)
                
                    if not os.path.exists(lq_path):
                        raise FileNotFoundError(f"{filename} not found in LQ folder")
                
                    self.paths.append({
                        'gt_path': gt_path,
                        'lq_path': lq_path
                    })
        else:
            # file client (lmdb io backend)
            if self.io_backend_opt['type'] == 'lmdb':
                self.io_backend_opt['db_paths'] = [self.gt_folder]
                self.io_backend_opt['client_keys'] = ['gt']
                if not self.gt_folder.endswith('.lmdb'):
                    raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
                with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                    self.paths = [line.split('.')[0] for line in fin]
            elif 'meta_info' in self.opt and self.opt['meta_info'] is not None:
                # disk backend with meta_info
                # Each line in the meta_info describes the relative path to an image
                with open(self.opt['meta_info']) as fin:
                    paths = [line.strip().split(' ')[0] for line in fin]
                    self.paths = [os.path.join(self.gt_folder, v) for v in paths]
            else:
                self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if opt['mode'] == 'train':
            # blur settings for the first degradation
            self.blur_kernel_size = opt['blur_kernel_size']
            self.kernel_list = opt['kernel_list']
            self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
            self.blur_sigma = opt['blur_sigma']
            self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
            self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
            self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

            # blur settings for the second degradation
            self.blur_kernel_size2 = opt['blur_kernel_size2']
            self.kernel_list2 = opt['kernel_list2']
            self.kernel_prob2 = opt['kernel_prob2']
            self.blur_sigma2 = opt['blur_sigma2']
            self.betag_range2 = opt['betag_range2']
            self.betap_range2 = opt['betap_range2']
            self.sinc_prob2 = opt['sinc_prob2']

            # a final sinc filter
            self.final_sinc_prob = opt['final_sinc_prob']

            self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
            # TODO: kernel range is now hard-coded, should be in the configure file
            self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
            self.pulse_tensor[10, 10] = 1

        # condition (independent of gt_size)
        if self.opt.get('condition_type', None) is not None:
            h = self.opt['gt_h'] // self.opt['scale']
            w = self.opt['gt_w'] // self.opt['scale']
            self.glob_condition = get_condition(h, w, self.opt['condition_type'])
        else:
            self.glob_condition = None
        
        self.phase = opt['mode']
        self.use_perspective = opt.get('use_perspective', True)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------- load GT -------------------- #
        if self.phase == 'train':
            gt_path = self.paths[index]
        else:
            gt_path = self.paths[index]['gt_path']
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
                break
            except (IOError, OSError):
                index = random.randint(0, self.__len__() - 1)
                gt_path = self.paths[index]
                time.sleep(1)
                retry -= 1

        img_gt = imfrombytes(img_bytes, float32=True)  # HWC, BGR, [0,1]

        if self.phase != 'train':
            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)

        # -------------------- augmentation (train only) -------------------- #
        if self.phase == 'train':
            img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        # -------------------- init perspective / condition -------------------- #
        _perspective = None
        _condition = None

        scale = self.opt['scale']
        gt_size = self.opt.get('gt_size', None)

        # ==================================================
        # TRAIN: random crop + valid perspective
        # ==================================================
        if self.phase == 'train' and gt_size is not None:
            h, w = img_gt.shape[:2]

            # pad if too small
            if h < gt_size or w < gt_size:
                pad_h = max(0, gt_size - h)
                pad_w = max(0, gt_size - w)
                img_gt = cv2.copyMakeBorder(
                    img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
                )

            h, w = img_gt.shape[:2]
            top = random.randint(0, h - gt_size)
            left = random.randint(0, w - gt_size)
            img_gt = img_gt[top:top + gt_size, left:left + gt_size, ...]

            # ---- perspective (LQ coordinate) ----
            top_lq = top // scale
            left_lq = left // scale
            _perspective = torch.tensor([top_lq, left_lq], dtype=torch.long)

            # ---- condition ----
            if self.opt.get('condition_type', None) is not None:
                _condition = self.glob_condition[
                    :, top_lq:top_lq + gt_size // scale,
                       left_lq:left_lq + gt_size // scale
                ]

        # ==================================================
        # VAL / TEST: full image + center perspective
        # ==================================================
        else:
            h, w = img_gt.shape[:2]

            if self.opt.get('use_perspective', True):
                top_lq = 0
                left_lq = 0
                _perspective = torch.tensor([top_lq, left_lq], dtype=torch.long)

            if self.opt.get('condition_type', None) is not None:
                _condition = self.glob_condition

        if self.phase == 'train':
            # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.opt['sinc_prob']:
                # this sinc filter setting is for kernels ranging from [7, 21]
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel = random_mixed_kernels(
                    self.kernel_list,
                    self.kernel_prob,
                    kernel_size,
                    self.blur_sigma,
                    self.blur_sigma, [-math.pi, math.pi],
                    self.betag_range,
                    self.betap_range,
                    noise_range=None)
            # pad kernel
            pad_size = (21 - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

            # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.opt['sinc_prob2']:
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel2 = random_mixed_kernels(
                    self.kernel_list2,
                    self.kernel_prob2,
                    kernel_size,
                    self.blur_sigma2,
                    self.blur_sigma2, [-math.pi, math.pi],
                    self.betag_range2,
                    self.betap_range2,
                    noise_range=None)

            # pad kernel
            pad_size = (21 - kernel_size) // 2
            kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

            # ------------------------------------- the final sinc kernel ------------------------------------- #
            if np.random.uniform() < self.opt['final_sinc_prob']:
                kernel_size = random.choice(self.kernel_range)
                omega_c = np.random.uniform(np.pi / 3, np.pi)
                sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
                sinc_kernel = torch.FloatTensor(sinc_kernel)
            else:
                sinc_kernel = self.pulse_tensor

            kernel = torch.FloatTensor(kernel)
            kernel2 = torch.FloatTensor(kernel2)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        if self.phase == 'train':
            return {
                'gt': img_gt, 
                'kernel1': kernel, 
                'kernel2': kernel2, 
                'sinc_kernel': sinc_kernel, 
                'gt_path': gt_path, 
                'condition': _condition, 
                'perspective': _perspective
            }

        else:
            return {
                'lq': img_lq, 
                'gt': img_gt, 
                'lq_path': lq_path, 
                'gt_path': gt_path, 
                'condition': _condition,
                'perspective': _perspective
            }


    def __len__(self):
        return len(self.paths)
    
def get_condition(h, w, condition_type):
    if condition_type is None:
        return 0.
    elif condition_type == 'cos_latitude':
        return torch.cos(make_coord([h]).unsqueeze(1).repeat([1, w, 1]).permute(2,0,1) * math.pi / 2)
    elif condition_type == 'latitude':
        return make_coord([h]).unsqueeze(1).repeat([1, w, 1]).permute(2, 0, 1) * math.pi / 2
    elif condition_type == 'coord':
        return make_coord([h, w]).permute(2, 0, 1)
    else:
        raise RuntimeError('Unsupported condition type')


def make_coord(shape, ranges=(-1, 1), flatten=False):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        v0, v1 = ranges
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret