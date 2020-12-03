import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class DIV2K(srdata.SRData):
    def __init__(self, args, train=True):
        super(DIV2K, self).__init__(args, train)
        self.repeat = args.test_every // (args.n_train // args.batch_size)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        if self.train:
            idx_begin = 0
            idx_end = self.args.n_train
        else:
            idx_begin = self.args.n_train
            idx_end = self.args.offset_val + self.args.n_val

        filenames = os.listdir(self.dir_hr)
        idx_end = len(filenames)
        for i in range(idx_begin, idx_end):
        # for filename in os.listdir(self.dir_hr):
            # filename = '{:0>4}'.format(i)
            filename = filenames[i]
            list_hr.append(os.path.join(self.dir_hr, filename))
            # list_hr.append(os.path.join(self.dir_hr, filename))
            for si, s in enumerate(self.scale):
            #     list_lr[si].append(os.path.join(
            #         self.dir_lr,
            #         'X{}/{}x{}{}'.format(s, filename, s, self.ext)
            #     ))
                list_lr[si].append(os.path.join(self.dir_lr, filename))

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = dir_data + '/train_1247'
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR')
        self.ext = '.jpg'

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.npy'.format(self.split)
        )

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.npy'.format(self.split, scale)
        )

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

