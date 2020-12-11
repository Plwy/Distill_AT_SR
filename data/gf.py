import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc
import imageio

import torch
import torch.utils.data as data

class GF(data.Dataset):
    def __init__(self, args, train=False):
        self._set_filesystem(args.dir_data)
        self.args = args
        self.filelist = []  # 图像路径
        self.idx_scale = 0

        if not train:
            for f in os.listdir(self.dir_lr):
                try:
                    filename = os.path.join(self.dir_lr, f)
                    self.filelist.append(filename)
                except:
                    pass

    def __getitem__(self, idx):
        filename = os.path.split(self.filelist[idx])[-1]
        filename, _ = os.path.splitext(filename)

        lr = imageio.imread(self.filelist[idx])
        lr = common.set_channel([lr], self.args.n_colors)[0]
        lr_tensor = common.np2Tensor([lr], self.args.rgb_range)[0]

        return lr_tensor, -1, filename


    def __len__(self):
        return len(self.filelist)

    def _set_filesystem(self, dir_data):
        # val_ocr_gaofa 测试
        self.dir_lr = os.path.join(dir_data, 'OCR/val_ocr_gaofa/image')
        self.ext = '.png'

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
