# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np
import math
# 2020/3/30 添加densenet:basic_module

#############2020/3/30:densenet:basic_module####################
# bottleneck
class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)
    def forward(self, x):
        out_conv1 = self.conv1(F.relu(self.bn1(x)))
        out_conv2 = self.conv2(F.relu(self.bn2(out_conv1)))
        out = torch.cat((x, out_conv2), 1)
        return out

# single
class SingleLayer(nn.Module):
    def __init__(self, nChannels, growRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growRate, kernel_size=3, padding=1, bias=False)
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

# avg_pool下采样
class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out

# conv下采样
class TransitionConv(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(TransitionConv, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(nOutChannels, nOutChannels, kernel_size=3, stride=2, padding=1, bias=False)
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(out)
        return out


# densenet--model
# nClasses是分类任务的类别数
class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()
        nDenseBlocks = (depth-4)//3

        if bottleneck:
            nDenseBlocks //= 2

        ###
        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False),
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck),
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels),

        ###
        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck),
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels),

        ###
        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

    ###
    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)

        return out

######################################################

# 不做16倍降采样处理：
class hourglass_no16(nn.Module):
    def __init__(self, in_channels):  # if==32
        super(hourglass_no16, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels, 3, 1, 1),  # 64--1/8
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels, in_channels, 3, 1, 1),  # 64--1/8
                                   nn.ReLU(inplace=True))

        # self.conv3 = nn.Sequential(convbn_3d(in_channels, in_channels, 3, 1, 1),  # 64--1/8
        #                            nn.ReLU(inplace=True))

        # self.conv4 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),  # 64--1/8
        #                            nn.ReLU(inplace=True))

        # self.conv5 = nn.Sequential(
        #     nn.ConvTranspose3d(in_channels * 2, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
        #     nn.BatchNorm3d(in_channels * 2))  # 64--1/8

        # self.conv5 = nn.Sequential(
        #     nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
        #     nn.BatchNorm3d(in_channels))  # 32--1/4

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))  # 32--1/4

        # self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        # self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        # conv3 = self.conv3(conv2)
        # conv4 = self.conv4(conv1 + conv3)  # 1/8

        # conv5 = F.relu(self.conv5(conv2), inplace=True)  # 1/4
        conv6 = F.relu(self.conv6(conv2), inplace=True)  # 1/4

        return conv6


'''

class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),  # 64--1/8
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)  # 64--1/8

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),  # 64--1/16
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),  # 64--1/16
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post
'''

# unet basic module

# 两个conv级联, no-res
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, 3, padding=1),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_ch, out_ch, 3, padding=1),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True)
        # )
        self.conv1 = convbn(in_ch, out_ch, 3, 1, 1, 1)
        self.conv2 = convbn(out_ch, out_ch, 3, 1, 1, 1)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(out1))
        return out2


# mobileNet-V2 basic module
# MobileBlock_1 (1*1通道升维+3*3+1*1+add), in_size = out_size
# MobileBlock_1(32, 32, 1, 1, 0, 1),
class MobileBlock_1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1):
        super(MobileBlock_1, self).__init__()
        self.in_channels_1 = int(in_channels * 2)  # 两倍升维
        self.conv1 = nn.Sequential(convbn(in_channels, self.in_channels_1, kernel_size, stride, padding, dilation), nn.ReLU(inplace=True))
        self.bn1 = nn.BatchNorm2d(self.in_channels_1)
        self.relu1 = nn.ReLU(inplace=True)
        self.depthwise = nn.Conv2d(self.in_channels_1, self.in_channels_1, 3, 1, 1, 1, groups=self.in_channels_1, bias=False)
        self.pointwise = nn.Conv2d(self.in_channels_1, out_channels, 1, 1, 0, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # ori_x = x  # add : need in_channels=out_channels
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.depthwise(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pointwise(out)
        out = self.bn2(out)
        # out = out + ori_x # add : need in_channels=out_channels
        return out

# MobileBlock_2 (1*1通道升维+3*3+1*1), 降采样1/2, in_size/2 = out_size
# MobileBlock_2(32, 64, 1, 1, 0, 1)
class MobileBlock_2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1):
        super(MobileBlock_2, self).__init__()
        self.in_channels_1 = int(in_channels * 2)  # 两倍升维
        self.conv1 = nn.Sequential(convbn(in_channels, self.in_channels_1, kernel_size, stride, padding, dilation), nn.ReLU(inplace=True))
        self.bn1 = nn.BatchNorm2d(self.in_channels_1)
        self.relu1 = nn.ReLU(inplace=True)
        self.depthwise = nn.Conv2d(self.in_channels_1, self.in_channels_1, 3, 2, 1, 1, groups=self.in_channels_1, bias=False)  # h/2, w/2
        self.pointwise = nn.Conv2d(self.in_channels_1, out_channels, 1, 1, 0, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.depthwise(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pointwise(out)
        out = self.bn2(out)
        return out

# 上采样 in_size *2 = out_size

class MobileBlock_up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1):
        super(MobileBlock_up, self).__init__()
        self.in_channels_1 = int(in_channels)  # 两倍升维
        self.conv1 = nn.Sequential(convbn(in_channels, self.in_channels_1, kernel_size, stride, padding, dilation), nn.ReLU(inplace=True))
        self.bn1 = nn.BatchNorm2d(self.in_channels_1)
        self.relu1 = nn.ReLU(inplace=True)
        self.depthwise = nn.ConvTranspose2d(self.in_channels_1, self.in_channels_1, 3, 2, 1, output_padding=1, groups=self.in_channels_1, bias=False, dilation=1)  # h*2, w*2
        self.pointwise = nn.Conv2d(self.in_channels_1, out_channels, 1, 1, 0, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.depthwise(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pointwise(out)
        out = self.bn2(out)
        return out