# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn import functional as F

#  pixel-wise loss  
# input shape: [N, C, H, W] ---> output: float
def pixelWiseLoss(preds_S, preds_T):
    preds_T = preds_T.detach()
    # print(preds_S.requires_grad, preds_T.requires_grad)
    assert preds_S.shape == preds_T.shape, 'the output dim of teacher and student differ'
    N, C, H, W = preds_S.shape
    softmax_pred_T = F.softmax(preds_T.permute(0, 2, 3, 1).contiguous().view(-1, C), dim=1)
    logsoftmax = nn.LogSoftmax(dim=1)
    loss = (torch.sum(- softmax_pred_T * logsoftmax(preds_S.permute(0, 2, 3, 1).contiguous().view(-1, C)))) / W / H
    return loss / N


# input shape: [N, C, H, W] ---> output: float
def KLpixelWiseLoss(preds_S, preds_T):
    preds_T = preds_T.detach()
    # print(preds_S.requires_grad, preds_T.requires_grad)  # RuntimeError: the derivative for 'target' is not implemented
    preds_S_logsoft = nn.functional.log_softmax(preds_S, dim=1)
    preds_T_soft = nn.functional.softmax(preds_T, dim=1)
    loss = nn.functional.kl_div(preds_S_logsoft, preds_T_soft)  # 默认为 reduction='mean' ???
    return loss


# feature similarity
def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0], 1, f_.shape[2], f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis

# pair-wise loss
def pairWiseLoss(preds_S, preds_T):
    scale = 0.5
    feat_S = preds_S[self.feat_ind]  # student
    feat_T = preds_T[self.feat_ind]  # teacher
    feat_T.detach()

    total_w, total_h = feat_T.shape[2], feat_T.shape[3]
    patch_w, patch_h = int(total_w * scale), int(total_h * scale)  # 类似于论文中的参数b，b个node形成一个团簇当做是一个新node
    # 无参
    maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True)  # change
    loss = sim_dis_compute(maxpool(feat_S), maxpool(feat_T))
    return loss


