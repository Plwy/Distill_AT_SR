
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
    log = torch.Tensor()
    p = [1,2,3]
    print(type(log))

    for i, k in enumerate(p):
        log[-1, i] += k
    
    print(type(log))
    print(log)


def test_feature_tran():
    #input size torch.Size([1, 3, 736, 1280]) 
    # tran to 
    x0 = np.random.randint(0,255,(5, 128, 3,3))
    x0 = torch.from_numpy(x0).float()  
    x1 = np.random.randint(0,255,(5, 1, 23, 12))
    x1 = torch.from_numpy(x1).float()  
    print(x0.shape)

    # print(x1.shape)

    x3 = x0.pow(2).mean(1)

    x3_1 = torch.mean(x0, 1, keepdim=True)
    x4 = F.normalize(x3)

    print(x3.shape)
    print(x4.shape)
    print(x3_1.shape) 

    # x2 = F.interpolate(x0, 
    #                 size=(x1.shape[2], x1.shape[3]), 
    #                 mode='bilinear', align_corners=True)

    # print(x2.shape)


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


def test_similarity_loss():

    x0 = np.random.randint(0,255,(5, 1, 12, 12))
    x0 = torch.from_numpy(x0).float()  

    x1 = np.random.randint(0,255,(5, 4, 24, 24))
    x1 = torch.from_numpy(x1).float()  

    loss = similarity_loss(x0, x1)

    print(loss)


def similarity_loss(f_s, f_t):
    bsz = f_s.shape[0]
    f_s = f_s.view(bsz, -1)     #shape(b,c*w*h)
    f_t = f_t.view(bsz, -1)     #shape(b,c*w*h)

    G_s = torch.mm(f_s, torch.t(f_s))   #(5,5)
    # G_s = G_s / G_s.norm(2)
    G_s = torch.nn.functional.normalize(G_s)   #
    G_t = torch.mm(f_t, torch.t(f_t))       #(5,5)
    # G_t = G_t / G_t.norm(2)
    G_t = torch.nn.functional.normalize(G_t)

    G_diff = G_t - G_s  
    print(G_diff.shape)
    print(G_diff)

    print(torch.abs(G_diff))
    print(G_diff * G_diff)

    # for x in G_diff:
    #     print(x)
    #     for i in x :
    #         if i < 0 :
    #             print("****")                
    loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
    return loss



if __name__ == '__main__':
    # test3()
    # test_feature_tran()
    # test_at()
    test_similarity_loss()
