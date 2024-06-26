import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    def __init__(self, args, train=True):
        super(Benchmark, self).__init__(args, train, benchmark=True)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        for entry in os.scandir(self.dir_hr):
            filename = os.path.splitext(entry.name)[0]
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            for si, s in enumerate(self.scale):
                # list_lr[si].append(os.path.join(
                #     self.dir_lr,
                #     'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                # ))
                print("test_path:", os.path.join(self.dir_lr, filename+self.ext))
                list_lr[si].append(os.path.join(self.dir_lr, filename+self.ext))

        list_hr.sort()
        for l in list_lr:
            l.sort()

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        # val_smimg_549验证
        self.apath = os.path.join(dir_data, 'val_smimg_549')
        self.dir_hr = os.path.join(self.apath, 'hr_image')
        self.dir_lr = os.path.join(self.apath, 'lr_image_x2_noise')
        self.ext = '.png'


        # test_set_2412测试
        # self.apath = os.path.join(dir_data, 'test_set_2412')
        # self.dir_hr = os.path.join(self.apath, 'hr_image')
        # self.dir_lr = os.path.join(self.apath, 'lr_image_x2_noise')
        # self.ext = '.jpg'