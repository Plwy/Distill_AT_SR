import numpy as np
import cv2
import torch
import os

import data

from collections import OrderedDict

from options import args
from utils import utility
from models import Model
from trainer import Trainer
from loss import Loss

from models.craft import CRAFT
# from models.rcan_at import RCAN
from models.rcan import RCAN


torch.manual_seed(args.seed) #为CPU设置种子用于生成随机数，以使得结果是确定的
# os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu_id)
# device = torch.device('cpu' if args.cpu else 'cuda')  


def test_teacher_model(net, image_path):
    """
        测试教师模型是否加载成功
    """
    from torch.autograd import Variable
    from utils import imgproc
    from Test_craft import feature_process

    # preprocessing
    image = imgproc.loadImage(image_path)
    # img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, 
    #                                                                     interpolation=cv2.INTER_LINEAR, 
    #                                                                     mag_ratio=args.mag_ratio)
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, 
                                                                        interpolation=cv2.INTER_LINEAR, 
                                                                        mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if not args.cpu:
        x = x.cuda()    

    with torch.no_grad():
        y, feature_ms = net(x)   # <class 'torch.Tensor'>,[1, 368, 640, 2],[1, 32, 368, 640]

    print(len(feature_ms))
    for f in feature_ms:
        print(type(f))
        print(f.shape)

        feature = feature_process(f)
        print(feature.shape)
        cv2.imshow('feature_img', feature)
        cv2.waitKey(0)   


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def load_teacher_model():
    teacher_net = CRAFT()
    print('Loading Teacher Model CRAFT',)
    if args.cpu:
        teacher_net.load_state_dict(copyStateDict(torch.load(args.teacher_ckpts, map_location='cpu')))
    else:
        teacher_net.load_state_dict(copyStateDict(torch.load(args.teacher_ckpts)))
        teacher_net = teacher_net.cuda()

    teacher_net.eval()

    if args.precision == 'half':
        teacher_net.half()

    for p in teacher_net.parameters():
        p.requires_grad = False

    return teacher_net


def train():
    # 训练数据加载
    loader = data.Data(args)  #<class 'data.Data'>
    teacher_model = load_teacher_model()  #<class 'models.craft.CRAFT'>
    # # test_teacher_model(teacher_model, 'Demo/TD/img_2.jpg')  

    student_ckpt = utility.checkpoint(args)   
    student_model = Model(args, student_ckpt)

    loss = Loss(args, student_ckpt) if not args.test_only else None
    trainer = Trainer(args, loader, student_model, teacher_model,  loss, student_ckpt) 

    while not trainer.terminate():
        trainer.train()
        trainer.test()   

    student_ckpt.done()

if __name__ == '__main__':
    train()