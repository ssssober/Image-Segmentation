# -*- coding: utf-8 -*-
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_module import *


# unet_model
class unet_model(nn.Module):
    def __init__(self, out_ch):
        super(unet_model, self).__init__()
        # self.conv1 = DoubleConv(3, 32)
        self.conv1 = convbn(3, 32, 3, 1, 1, 1)  # (32, h, w)
        self.pool1 = nn.MaxPool2d(2)  # (32, h/2, w/2)
        self.conv2 = DoubleConv(32, 64)  # (64, h/2, w/2)
        self.pool2 = nn.MaxPool2d(2)  # (64, h/4, w/4)
        self.conv3 = DoubleConv(64, 128)  # (128, h/4, w/4)
        self.pool3 = nn.MaxPool2d(2)  # (128, h/8, w/8)
        self.conv4 = DoubleConv(128, 256)  # (256, h/8, w/8)
        self.pool4 = nn.MaxPool2d(2)  # (256, h/16, w/16)
        self.conv5 = DoubleConv(256, 128)  # (128, h/16, w/16)

        self.up6 = nn.ConvTranspose2d(128, 128, 2, stride=2)  # (128, h/8, w/8)
        # self.conv6 = DoubleConv(384, 128)  # up6+c4
        self.conv6 = convbn(384, 128, 3, 1, 1, 1)  # (128, h/8, w/8)
        self.up7 = nn.ConvTranspose2d(128, 128, 2, stride=2)  # (128, h/4, w/4)
        self.conv7 = DoubleConv(256, 128)  # up7+c3

        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # (64, h/2, w/2)
        self.conv8 = DoubleConv(128, 64)  # up8+c2

        self.up9 = nn.ConvTranspose2d(64, 64, 2, stride=2)  # (64, h, w)

        # self.conv9 = DoubleConv(96, 32)  # up9+c1
        self.conv9 = convbn(96, 32, 3, 1, 1, 1)
        self.conv10 = nn.Conv2d(32, out_ch, 1)  # (out_ch, h, w)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        c1 = self.conv1(x)  # [3, 64]
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)  # [64, 128]
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)  # [128, 256]
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)  # (256, 256)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c9 = F.relu(c9)
        c10 = self.conv10(c9)
        # out = nn.Sigmoid()(c10)
        # return out
        return c10
