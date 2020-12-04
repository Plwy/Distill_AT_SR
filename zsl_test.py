import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def main():
    f = np.random.randint(0,255,(5,1,3,4))

    f = torch.from_numpy(f).float()
    print(f)

    print(f.dtype)
    f_n = F.normalize(f)
    print(f_n)
    s = nn.Sigmoid()
    f_s = s(f_n)
    print(f_s)


if __name__ == "__main__":

    main()