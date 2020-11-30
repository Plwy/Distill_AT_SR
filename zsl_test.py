
import numpy as np
import torch
import cv2
from options import args
import torch.nn.functional as F
from utils.at_utils import *

def test():

    for d in args.data_train:
        print(d)
        # module_name = d if d.find('DIV2K') == 0 else 'DIV2KJPEG'
        module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'

        print(module_name)

    args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    print(args.scale)

# def test2():
#     x = 
#     F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def test3():
    x1 = np.random.randint(0,10,(1,3,4,5))
    x1 = torch.from_numpy(x1).float()
    x1_m = torch.mean(x1, dim=1, keepdim=True)

    x2 = np.random.randint(0,10,(1,3,4,5))
    x2 = torch.from_numpy(x2).float()
    x2_m = torch.mean(x2, dim=1, keepdim=True)

    x3 = torch.cat([x1_m, x2_m], dim=1)

    print(x1.shape)
    print(x1_m.shape)
    print(x2_m.shape)
    print(x3.shape)

    # student 空间注意力 结束后输出(1,1, h,w)

    sg = torch.nn.Sigmoid()
    x4 = sg(x3)
    print(x4.shape)

def test_feature_tran():
    #input size torch.Size([1, 3, 736, 1280]) 
    # tran to 
    x0 = np.random.randint(0,255,(5, 128, 3,3))
    x0 = torch.from_numpy(x0).float()  
    x1 = np.random.randint(0,255,(5, 1, 23, 12))
    x1 = torch.from_numpy(x1).float()  
    print(x0.shape)

    print(x1.shape)
    x2 = F.interpolate(x0, 
                    size=(x1.shape[2], x1.shape[3]), 
                    mode='bilinear', align_corners=True)

    print(x2.shape)


def test_at():
    
    x1 = np.random.randint(0,255,(5, 4, 9, 9))
    x1 = torch.from_numpy(x1).float()  

    x0 = np.random.randint(0,255,(5, 1, 9,9))
    x0 = torch.from_numpy(x0).float()  

    # x3 = x1.expand(1,2,2,3)
    # print(x3.shape)

    x1_a = activate_spacial_attention(x1)
    x0_a = activate_spacial_attention(x0)

    print(x1_a.shape)
    print(x0_a.shape)

    l = (x0_a - x1_a).pow(2).mean()
    print(l)

    import torch.nn as nn
    l = nn.MSELoss()
    loss = l(x0_a, x1_a)
    print(loss)
    print("%.3f"%loss) 



def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()



if __name__ == '__main__':
    # test3()
    test_feature_tran()
    # test_at()
