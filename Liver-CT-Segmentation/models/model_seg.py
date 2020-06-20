# -*- coding: utf-8 -*-
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv1 = convbn(in_ch, out_ch, 3, 1, 1, 1)
        self.conv2 = convbn(out_ch, out_ch, 3, 1, 1, 1)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(out1))
        return out2

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))

class seg_model(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(seg_model, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.down_1 = convbn(64, 64, 3, 2, 1, 1),
        self.conv2 = DoubleConv(64, 128)
        self.down_2 = convbn(128, 128, 3, 2, 1, 1),
        self.conv3 = DoubleConv(128, 256)
        self.down_3 = convbn(256, 256, 3, 2, 1, 1),
        self.conv4 = DoubleConv(256, 512)
        self.down_4 = convbn(512, 512, 3, 2, 1, 1),

        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

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
        c1 = self.conv1(x)
        d1 = self.down_1(c1)
        c2 = self.conv2(d1)
        d2 = self.down_2(c2)
        c3 = self.conv3(d2)
        d3 = self.down_3(c3)
        c4 = self.conv4(d3)
        d4 = self.down_1(c4)
        c5 = self.conv5(d4)
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
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out
