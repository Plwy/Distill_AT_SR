import models.common as common
import torch
import torch.nn as nn

def make_model(args, parent=False):
    return RCAN(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

"""
    Additional Attention
    from https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
"""
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RCAN, self).__init__()
        self.args = args
        
        self.n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # redefine body module
        ###################
        for group_id in range(self.n_resgroups):
            setattr(self, 'body_group{}'.format(str(group_id)), 
                    ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, 
                                    res_scale=args.res_scale, n_resblocks=n_resblocks))
        
        self.body_tail = conv(n_feats, n_feats, kernel_size)

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, sign=1)

        self.head = nn.Sequential(*modules_head)
        self.tail = nn.Sequential(*modules_tail)
        
        # SA
        self.sa = SpatialAttention()

    def forward(self, x):
        # featuremaps记录空间注意力特征图 
        fms = []    

        x = self.sub_mean(x)
        x = self.head(x)
        
        # sa_map 1 从head层提取
        sa_map1 = self.sa(x) 
        fms.append(sa_map1)
        x = sa_map1 * x

        # sa_map 2,3 从RG中提取
        res = x
        for group_id in range(self.n_resgroups):
            res = getattr(self, 'body_group{}'.format(str(group_id)))(res)
            if group_id == 2 or group_id == 6:
                rg_sa_map = self.sa(res)
                res = rg_sa_map * res
                fms.append(rg_sa_map)

        # samap 4 从RG层结束获取 
        res = self.body_tail(res)
        sa_map4 = self.sa(res)
        fms.append(sa_map4)
        res = sa_map4 * res
        
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x, fms
     
    def load_state_dict_teacher(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            old_name = name
            if 'body' in name:
                a = name.split('.')
                if int(a[1]) == 10:
                    a[0] = 'body_tail'
                else:
                    a[0] = 'body_group' + a[1]
                a.pop(1)
                name = '.'.join(a)
        
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                own_state[name].copy_(param)
            else:
                print(name, old_name)

                
    def load_state_dict_student(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                own_state[name].copy_(param)
            else:
                print(name)    
