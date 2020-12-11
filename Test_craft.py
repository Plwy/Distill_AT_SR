# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from PIL import Image
# from skimage import io
# import json
# import zipfile

from utils import imgproc
from utils import craft_utils
from utils import file_utils

from models.craft import CRAFT
from collections import OrderedDict

"""
    输入单张图片，输出中间特征图
"""
def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

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

def arg_parser():
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_model', default='ckpts/teacher_ckpts/CRAFT_model/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default='test_data', type=str, help='folder path to input images')
    parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default='weights/CRAFT_model/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

    args = parser.parse_args()

    return args

def test_craft(image_path):
    args = arg_parser()
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    net.eval()

    # preprocessing
    image = imgproc.loadImage(image_path)
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, 
                                                                        interpolation=cv2.INTER_LINEAR, 
                                                                        mag_ratio=args.mag_ratio)
    print(image.shape)
    print(img_resized.shape)
    # print('tr', target_ratio)
    # print(size_heatmap)

    ratio_h = ratio_w = 1 / target_ratio
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if args.cuda:
        x = x.cuda()    

    with torch.no_grad():
        y, feature_ms = net(x)   # <class 'torch.Tensor'>,[1, 368, 640, 2],[1, 32, 368, 640]

    print(len(feature_ms))
    for f in feature_ms:
        feature = feature_process(f)
        cv2.imshow('feature_img', feature)
        cv2.waitKey(0)
    """
        网络输出后处理
    """
    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy() # 文本区域特征
    score_link = y[0,:,:,1].cpu().data.numpy() # 相互关联得分

    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, args.text_threshold, args.link_threshold, args.low_text, args.poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    result_folder = './TextDe_result/'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    return y     

def draw_heatmap(img, f):
    def cvt2HeatmapImg(img):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        return img
    
    



def feature_process(feature):
    print('before feature shape', feature.shape)  #torch.Size([1, 32, 368, 640])
    feature=feature[:,0,:,:]
    feature=feature.view(feature.shape[1],feature.shape[2])

    feature=feature.data.cpu().numpy()

    feature= 1.0/(1+np.exp(-1*feature))        #use sigmod to [0,1]
    feature=np.round(feature*255)   # to [0,255]
    feature=np.array(feature, dtype=np.uint8)
    print("output feature:",feature.shape, feature.dtype)
    return feature


def main():
    image_path = 'Demo/TD/img_2.jpg' # 待检测图片路径
    y = test_craft(image_path)


if __name__ == '__main__':
    main()