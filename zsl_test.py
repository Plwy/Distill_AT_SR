
import numpy as np
import torch
import cv2
from options import args
import torch.nn.functional as F

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

def AT(fm):
    eps=1e-6
    am = torch.pow(torch.abs(fm), 2)
    am = torch.sum(am, dim=1, keepdim=True)
    norm = torch.norm(am, dim=(2,3), keepdim=True)
    am = torch.div(am, norm+eps)
    return am

def test_feature_tran():
    #input size torch.Size([1, 3, 736, 1280]) 
    # tran to 

    x1 = np.random.randint(0,255,(5, 3, 2,3))
    x1 = torch.from_numpy(x1).float()

    # x1 = torch.pow(torch.abs(x1), 2)
    # print(x1.shape)
    # x1_sum = torch.sum(x1, dim=1, keepdim=True)
    # print(x1_sum.shape)
    # print(x1_sum)

    # # norm = torch.norm(x1_sum, 2, dim=2, keepdim=True)


    # norm = F.normalize(x1_sum,p=1,dim=3)    #对指定维度进行运算

    # print(norm)
    # print(norm.shape)
    # m = norm.mean()
    # print("{:.2f}".format(m))

    x_2 = x1.pow(2).mean(1)
    x_3 = x_2.view(x1.size(0),-1)
    x_4 = F.normalize(x_3)
    print(x_2.shape)
    print(x_3.shape)
    print(x_3)
    print(x_4.shape)
    print(x_4)

if __name__ == '__main__':
    # test3()
    test_feature_tran()