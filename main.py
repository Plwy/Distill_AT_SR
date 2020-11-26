import numpy as np
import cv2
import torch
import os


from options import args
from models import *
from utils import utility
import data

torch.manual_seed(args.seed) #为CPU设置种子用于生成随机数，以使得结果是确定的
torch.cuda.manual_seed(args.seed) #为当前GPU设置随机种子；
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu_id)
device = torch.device('cpu' if args.cpu else 'cuda')  

def load_teacher_model():
    teacher_net = CRAFT()
    print('Loading Teacher Model CRAFT',)
    teacher_net.load_state_dict_teacher(torch.load(args.teacher_ckpts))

    if args.precision == 'half':
        teacher_net.half()

    for p in teacher.parameters():
        p.requires_grad = False
    return

def create_student_model():
    student_checkpoint = utility.checkpoint(args)
    student_net = rcan.RCAN(args).to(device)

    if args.precision == 'half':
        teacher_net.half()

    return student_checkpoint, student


def train():
    """
        训练
    """
    # 训练数据加载
    loader = data.Data(args)

    print(type(loader))


    # teacher_model = load_teacher_model()
    # student_ckpt, student_model = create_student_model()

    # trainer = Trainer(args, loader, student_model, loss, student_ckpt) 

    # if not args.test_only:
    #     trainer.train()
    #     trainer.test()
    # else:
    #     trainer.test()

    # student_ckpt.done()
    # 优化器
    


# def eval():
#     """
#         推导
#     """



if __name__ == '__main__':
    train()