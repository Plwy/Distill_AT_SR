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

        self.feature_loss = [] 
        self.feature_loss_module = nn.ModuleList()


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
                    'function': SDLoss(loss=nn.MSELoss())}   

            # self.loss.append(l)
            self.feature_loss.append(l)
            self.feature_loss_module.append(l['function'])                 
        

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        if len(self.feature_loss) > 1:
            self.feature_loss.append({'type': 'Total', 'weight': 0, 'function': None})

        # 添加loss模块
        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        print("===== feature functions ====")
        print(len(self.feature_loss_module))

        print("===== loss functions ====")
        print(len(self.loss_module))

        # 模块设置
        self.log = torch.Tensor()
        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        self.feature_loss_module.to(device)

        if args.precision == 'half': 
            self.loss_module.half()

        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )
            self.feature_loss_module = nn.DataParallel(
                self.feature_loss_module, range(args.n_GPUs)
            )

        if args.load != '.': self.load(ckp.dir, cpu=args.cpu)



    def forward(self, sr, hr, student_fms, teacher_fs):
        """
            L = w1*L_sr + w2*L_ad + w3*L_sd
        """
        # SR loss
        losses = []
        count = 0
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()        
                count += 1
        sr_loss_sum = sum(losses)

        # Activate distill loss, Similarity distill loss
        assert(len(student_fms) == len(teacher_fs))
        print("shape s_fms, t_fms", student_fms[0].shape, teacher_fs[0].shape)

        for i, l in enumerate(self.feature_loss):
            if l['function'] is not None:
                for j in range(len(student_fms)):
                    loss = l['function'](student_fms, teacher_fs)  # 输入为学生注意力特征图和教师特征
                    feature_loss += loss
                feature_loss =  l['weight'] * loss
                feature_losses.append(feature_loss)
                self.log[-1, i+count] += feature_losses.item()  

        print("feature_losses:", feature_losses)
        feature_loss_sum = sum(feature_losses)

        loss_sum = sr_loss_sum + feature_loss_sum
        
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
        axis = np.linspace(1, epoch, epoch)
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
            return self.feature_loss_module
        else:
            return self.feature_loss_module.module

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
        for l in self.feature_loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()


class ADLoss(nn.Module):
    """
        激活蒸馏损失
    """
    def __init__(self, loss=nn.MSELoss()):
        super(ADLoss, self).__init__()
        self.loss = loss

    def forward(self, fms_s, fs_t):
        # 输入为学生注意力特征图和教师特征
        assert len(fms_s) == len(fs_t)
        length = len(fms_s)

        # 根据教师特征提取空间注意力激活映射
        print("Teacher feature activate~~~~~~~~~")
        fms_t = []
        for f in fs_t:
            print(f.shape)
            # fms = F.normalize(f.pow(2).mean(1).view(f.size(0), -1))  # 转为 (b,w*h)向量形式
            fms = F.normalize(f.pow(2).mean(1))
            fms_t.append(f)
        print('after tran',fms_t[1].shape)

        # # 将学生网络的注意力转为向量形式
        # for i in range(len(fms_s)):
        #     fms_s[i] = fms_s[i].view(fms_s[i].size(0), -1)

        # 计算MSE

        adloss = [self.loss(fms_s[i], fms_t[i]) for i in range(length)]
        adloss_sum = sum(adloss)

        return adloss_sum


class SDLoss(nn.Module):
    """
        相似性矩阵蒸馏损失
        Frobenius范数：矩阵A的Frobenius范数定义为矩阵A各项元素的绝对值平方的总和开根
    """
    def __init__(self, loss=nn.MSELoss()):
        super(SDLoss, self).__init__()
        self.loss = loss

    def forward(self, fms_s, fs_t):

        assert len(fms_s) == len(fs_t)
        length = len(fms_s)

        # 根据教师特征提取空间注意力激活映射
        print("Teacher feature activate~~~~~~~~~")
        fms_t = []
        for f in fs_t:
            print(f.shape)
            # fms = F.normalize(f.pow(2).mean(1).view(f.size(0), -1))  # 转为 (b,w*h)向量形式
            fms = F.normalize(f.pow(2).mean(1))
            fms_t.append(f)

        # # 将学生网络的注意力转为向量形式
        # for i in range(len(fms_s)):
        #     fms_s[i] = fms_s[i].view(fms_s[i].size(0), -1)

        # input size (b,1,w,h)
        sdloss = [self.loss(fms_s[i], fms_t[i]) for i in range(length)]
        sdloss_sum = sum(sdloss)        

        return sdloss_sum


class FeatureLoss(nn.Module):
    def __init__(self, loss=nn.L1Loss()):
        super(FeatureLoss, self).__init__()
        self.loss = loss

    def forward(self, outputs, targets):
        assert len(outputs)
        assert len(outputs) == len(targets)
        length = len(outputs)
        tmp = [self.loss(outputs[i], targets[i]) for i in range(length)]
        loss = sum(tmp)
        return loss



