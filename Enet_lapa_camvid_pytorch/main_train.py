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

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='ENet-ori')
parser.add_argument('--mode', type=str, default='test', help='train or test')
parser.add_argument('--model', default='fd', help='select a model structure', choices=__models__.keys())
parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default='', help='data path')
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
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch begin train')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
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
model = nn.DataParallel(model)  #
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
# optimizer = optim.RMSprop(model.parameters(), lr=args.lr)


loss_back = nn.CrossEntropyLoss()  # 多分类:CrossEntropyLoss = log_softmax() + NLLLoss()
# loss_back = nn.NLLLoss()  # 

# output model parameters
print("Number of model parameters: {}".format(sum([p.data.nelement() for p in model.parameters()])))


# load parameters
# start_epoch = args.start_epoch
start_epoch = 0
if args.resume:  # 
    # find all checkpoints file and sort according to epoch id
    # all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    # all_saved_ckpts = [fn for fn in os.listdir(args.checkpoint_path) if fn.endswith(".tar")]
    # all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # # use the latest checkpoint file
    # loadckpt = os.path.join(args.checkpoint_path, all_saved_ckpts[-1])
    # print("loading the lastest model in logdir: {}".format(loadckpt))
    # state_dict = torch.load(loadckpt)
    # # model.load_state_dict(state_dict['model'])
    # model.load_state_dict(state_dict['state_dict'])
    # optimizer.load_state_dict(state_dict['optimizer'])
    # start_epoch = state_dict['epoch'] + 1
    # 指定被加载模型，继续训练
    print("loading the lastest model in logdir: {}".format(args.checkpoint_path))
    state_dict = torch.load(args.checkpoint_path)
    model.load_state_dict(state_dict['state_dict'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    # model.load_state_dict(state_dict['model'])
    model.load_state_dict(state_dict['state_dict'])
print("start at epoch {}".format(start_epoch))

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
    model.train()

    # training data load
    ori, gt = sample['ori'], sample['gt']

    groundtruth = gt
    groundtruth = groundtruth.cuda()
    ori = ori.cuda()
    gt = gt.cuda()
    optimizer.zero_grad()
    # 模型输出
    pre = model(ori)
    
    # pre_label = torch.max(pre, dim=1)  # 
    pred = nn.functional.softmax(pre, dim=1)  # 用于显示--测试只要softmax()，方式二正确
    pre_label = torch.max(pred, dim=1)
    loss = loss_back(pre, gt)
    scalar_outputs = {"loss": loss}

    image_outputs = {"ori_image": ori, "ground_truth": groundtruth, "pred": pre_label[1]}  # 取tuple[1]

    # # dict update 暂时保留，后续是否输出errormap
    # if compute_metrics:
    #     with torch.no_grad():
   
    #

    loss.backward()
    optimizer.step()
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs

'''
# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()

    imgL, imgR, d_gt = sample['l'], sample['r'], sample['d']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    d_gt = d_gt.cuda()

    

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs
'''

if __name__ == '__main__':
    if args.mode == 'train':
        train()