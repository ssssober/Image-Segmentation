# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
import gc
from utils.color_map import *
from models.loss import *

'''
蒸馏方法：--soft KD
teacher：unet
student: enet / mobileV2
loss: kl loss(s_pre, t_pre) + nn.CrossEntropyLoss(s_pre, gt)
training type：resume a student
'''

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='ENet-ori')
parser.add_argument('--mode', type=str, default='train', help='train or test')
parser.add_argument('--model', default='xxnet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--model_teacher', default='xxnet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default='', help='data path')
# parser.add_argument('--channels', type=int, default=1, help='dataloader input channels')
parser.add_argument('--in_channels', type=int, default=1, help='net input channels')
parser.add_argument('--out_channels', type=int, default=1, help='net output channels')
parser.add_argument('--trainlist', required=True, help='training list')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--train_crop_height', type=int, default=128, help='training crop height')
parser.add_argument('--train_crop_width', type=int, default=256, help='training crop width')
parser.add_argument('--lr', type=float, default=0.0001, help='base learning rate')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')
parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch begin train')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--loadteacher', action='store_true', help='load the weights from a specific checkpoint')
parser.add_argument('--ckpt_path_teacher', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--w_pi', type=float, default=1.0, help='weight of pixel loss')
# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True, args.train_crop_height, args.train_crop_width, args.in_channels)
# test_dataset = StereoDataset(args.datapath, args.testlist, False, args.test_crop_height, args.test_crop_width, args.channels)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=6, drop_last=True)
# TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

train_file = open(args.trainlist, "r")
train_file_lines = train_file.readlines()
print("train_file_lines nums: ", len(train_file_lines))

# model, optimizer
model = __models__[args.model](args.out_channels)
model = nn.DataParallel(model)  # 多卡
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
# optimizer = optim.RMSprop(model.parameters(), lr=args.lr)


loss_back = nn.CrossEntropyLoss()  # 多分类:CrossEntropyLoss = log_softmax() + NLLLoss()

print("Number of model parameters: {}".format(sum([p.data.nelement() for p in model.parameters()])))

# ## load teacher #
model_T = __models__[args.model_teacher](args.out_channels)
model_T = nn.DataParallel(model_T)
model_T.cuda()
print("Number of teacher model parameters: {}".format(sum([p.data.nelement() for p in model_T.parameters()])))
# ##

# load parameters
# start_epoch = args.start_epoch
start_epoch = 0
if args.resume:  # 继续训练
    print("loading the lastest model in logdir: {}".format(args.checkpoint_path))
    state_dict = torch.load(args.checkpoint_path)
    model.load_state_dict(state_dict['state_dict'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
    print("resume start at epoch {}".format(start_epoch))
if args.loadteacher:
    # load the checkpoint file specified by args.loadckpt
    print("loading teacher model {}".format(args.ckpt_path_teacher))
    state_dict = torch.load(args.ckpt_path_teacher)
    # model.load_state_dict(state_dict['model'])
    model_T.load_state_dict(state_dict['state_dict'])


def train():
    train_start_time = time.time()
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            # 屏幕打印
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary)
            if do_summary:  # 屏幕打印
                print('Epoch {}/{}, Iter {}/{}, Global_step {}/{}, train loss = {:.3f}, time = {:.3f}, time elapsed {:.3f}, time left {:.3f}h'.format(epoch_idx, args.epochs,
                batch_idx, len(TrainImgLoader), global_step, len(TrainImgLoader) * args.epochs, loss, time.time() - start_time, (time.time() - train_start_time) / 3600,
                                                                    (len(TrainImgLoader) * args.epochs / (global_step + 1) - 1) * (time.time() - train_start_time) / 3600))
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs

            '''
            # saving checkpoints(ckpt)
            if (epoch_idx + 1) % args.save_freq == 0:
                checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
            '''
            # saving checkpoints(tar)
            #if (global_step % args.save_freq == 0 and int(global_step / args.save_freq) != 0) or batch_idx + 1 == len(TrainImgLoader):
            if int(global_step / args.save_freq) != 0 and global_step % args.save_freq == 0:
                checkpoint_data = {'epoch': epoch_idx, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(checkpoint_data, "{}/checkpoint_{}_{:0>7}.tar".format(args.logdir, epoch_idx + 1, global_step))
        gc.collect()

        # # testing
        # avg_test_scalars = AverageMeterDict()
        # for batch_idx, sample in enumerate(TestImgLoader):
        #     global_step = len(TestImgLoader) * epoch_idx + batch_idx
        #     start_time = time.time()
        #     do_summary = global_step % args.summary_freq == 0
        #     loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
        #     if do_summary:
        #         save_scalars(logger, 'test', scalar_outputs, global_step)
        #         save_images(logger, 'test', image_outputs, global_step)
        #     avg_test_scalars.update(scalar_outputs)
        #     del scalar_outputs, image_outputs
        #     print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
        #                                                                              batch_idx,
        #                                                                              len(TestImgLoader), loss,
        #                                                                              time.time() - start_time))
        # avg_test_scalars = avg_test_scalars.mean()
        # save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        # print("avg_test_scalars", avg_test_scalars)
        # gc.collect()

# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()  # student
    model_T.eval()  # teacher
    ori, gt = sample['ori'], sample['gt']
    ori = ori.cuda()
    gt = gt.cuda()
    optimizer.zero_grad()

    # student :  train
    pre = model(ori)
    loss_s = loss_back(pre, gt)  # student : crossEntropy()  --> torch.cuda.FloatTensor
    pred = nn.functional.softmax(pre, dim=1)  # 
    # print(pred.size())  # (batch_size, out_classes, crop_h, crop_w)
    pre_label = torch.max(pred, dim=1)  # tuple
   
    # teacher: eval
    pre_T = model_T(ori)
    loss_T = loss_back(pre_T, gt)
    pred_T = nn.functional.softmax(pre_T, dim=1)
    pre_label_T = torch.max(pred_T, dim=1)  # tuple[max_value, index]
    pixel_loss = KLpixelWiseLoss(pre, pre_T)
    pixel_loss = args.w_pi * pixel_loss
    loss_sum = loss_s + pixel_loss
    # display()
    scalar_outputs = {"loss_s": loss_s, "loss_T": loss_T, "pixel_loss": pixel_loss, "loss_sum": loss_sum}
    image_outputs = {"ori_image": ori, "ground_truth": gt, "pred_s": pre_label[1], "pred_t": pre_label_T[1]}  # 
    loss_sum.backward()
    optimizer.step()
    return tensor2float(loss_sum), tensor2float(scalar_outputs), image_outputs

'''
# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    disp_ests = model(imgL, imgR)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    loss = multi_model_loss(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    if compute_metrics:
        image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs
'''

if __name__ == '__main__':
    if args.mode == 'train':
        train()