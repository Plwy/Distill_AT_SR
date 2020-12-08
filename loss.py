import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        """
            SR loss : 可以选用多种方式
        """
        super(Loss, self).__init__()
        print('Preparing loss function:')
        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()

        # 不同的SR loss计算
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )

            self.loss.append({'type': loss_type, 
                            'weight': float(weight),
                            'function': loss_function})

        # 添加蒸馏损失
        for loss in args.feature_distilation_type.split('+'):
            print("distill loss", loss)
            weight, feature_type = loss.split('*')
            
            # 空间注意力激活蒸馏损失
            if feature_type == 'AD':
                l = {'type': feature_type, 
                    'weight': float(weight), 
                    'function': ADLoss(loss=nn.MSELoss())}   

            # 相似性矩阵知识蒸馏损失   
            elif feature_type == 'SD':
                l = {'type': feature_type, 
                    'weight': float(weight), 
                    'function': SDLoss()}            
            self.loss.append(l)

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        # 添加loss模块
        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        # 模块设置
        self.log = torch.Tensor()
        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)

        if args.precision == 'half': 
            self.loss_module.half()

        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        if args.load != '.': self.load(ckp.dir, cpu=args.cpu)



    def forward(self, sr, hr, student_fms, teacher_fs):
        """
            L =（w_0*L0_sr+...+w_n*L1_sr）+ w_ad*L_ad + w_sa*L_sd
        """
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                if l['type'] in ['AD', 'SD']:
                    loss = l['function'](student_fms, teacher_fs)
                else:
                    loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()      
        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        print(losses)
        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch-1, epoch-1)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()


class ADLoss(nn.Module):
    """
        activate distill loss
    """
    def __init__(self, loss=nn.MSELoss()):
        super(ADLoss, self).__init__()
        self.loss = loss

    def forward(self, fms_s, fs_t):
        # 输入为学生注意力特征图list和教师特征list
        assert len(fms_s) == len(fs_t)
        length = len(fms_s)

        # 教师特征上采样 尺寸一致
        for i in range(len(fs_t)):
            fs_t[i] = F.interpolate(fs_t[i], 
                            size=(fms_s[i].shape[2], fms_s[i].shape[3]), 
                            mode='bilinear', align_corners=True)

        # 教师特征提取空间注意力激活映射
        fms_t = []
        for f in fs_t:
            fms = F.normalize(torch.mean(f.pow(2), 1, keepdim=True))  # 转为(b,1,w,h)
            fms_t.append(fms)

        # MSE计算注意力特征图的相似性蒸馏损失
        adloss = [self.loss(fms_s[i], fms_t[i]) for i in range(length)]
        adloss_sum = sum(adloss)

        return adloss_sum


class SDLoss(nn.Module):
    """
        similarity matrix distill loss
    """
    def __init__(self):
        super(SDLoss, self).__init__()

    def forward(self, fms_s, fs_t):
        # 相似性矩阵计算方法不考虑两个特征图的尺寸
        assert len(fms_s) == len(fs_t)
        length = len(fms_s)

        fms_t = []
        for f in fs_t:
            fms = F.normalize(torch.mean(f.pow(2), 1, keepdim=True))  # 转为(b,1,w,h)
            fms_t.append(f)

        sdloss = [self.similarity_loss(fms_s[i], fms_t[i]) for i in range(length)]
        sdloss_sum = sum(sdloss)        
        return sdloss_sum

    def similarity_loss(self, f_s, f_t):
        # same batch
        assert f_s.shape[0] == f_t.shape[0]      

        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)     #(b,c_s*w_s*h_s)
        f_t = f_t.view(bsz, -1)     #(b,c_t*w_t*h_t)

        G_s = torch.mm(f_s, torch.t(f_s))   #(b,b)
        G_s = torch.nn.functional.normalize(G_s)   
        G_t = torch.mm(f_t, torch.t(f_t))       
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s                  
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)

        return loss


        



