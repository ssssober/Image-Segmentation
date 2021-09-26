# -*- coding: utf-8 -*-
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines
from datasets.data_io import *
import torchvision.transforms as transforms
import torch
import torch.nn as nn


processed = transforms.Compose([transforms.ToTensor(),])


class LapaPngPng(Dataset):
    def __init__(self, datapath, list_filename, training, crop_h, crop_w, channel):
        self.datapath = datapath
        self.ori_filenames, self.gt_filenames = self.load_path(list_filename)
        self.training = training
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.channel = channel
        self.transform_img = transforms.Compose(
            [
                transforms.Resize(size=(self.crop_h, self.crop_w)),  # Resize：尺寸随意变大变小;  h, w
            ]
        )
        #  error:Floating point exception(core dumped)
        self.processed_color = transforms.Compose([transforms.ColorJitter(brightness=0.5, contrast=0.5),
                                    transforms.ToTensor(), #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        ori_images = [x[0] for x in splits]
        gt_images = [x[1] for x in splits]
        return ori_images, gt_images

    # load rgb-color
    def load_image(self, filename):
        if self.channel == 3:
            return Image.open(filename).convert('RGB')
        elif self.channel == 1:
            return Image.open(filename).convert('L')

    def load_img(self, filename):
        return Image.open(filename).convert('L')

    def __len__(self):
        return len(self.ori_filenames)

    # augmentation
    def augment_image_pair(self, left_image):
        # randomly shift gamma
        random_gamma = torch.rand(1).numpy()[0] * 0.4 + 0.8  # random.uniform(0.8, 1.2)
        left_image_aug = left_image ** random_gamma

        # randomly shift brightness
        random_brightness = torch.rand(1).numpy()[0] * 1.5 + 0.5  # random.uniform(0.5, 2.0)
        left_image_aug = left_image_aug * random_brightness

        # randomly shift color
        if self.channel == 3:
            random_colors = (torch.rand(1).numpy()[0] * 0.4 + 0.8, torch.rand(1).numpy()[0] * 0.4 + 0.8,
                             torch.rand(1).numpy()[0] * 0.4 + 0.8)
            white = torch.ones(left_image.shape[1], left_image.shape[2])
            color_image = torch.stack((white * random_colors[0], white * random_colors[1], white * random_colors[2]),dim=0)
            left_image_aug *= color_image

        # saturate
        left_image_aug = torch.clamp(left_image_aug, 0, 1)
        return left_image_aug



    def __getitem__(self, index):
        ori_img = self.load_image(os.path.join(self.datapath, self.ori_filenames[index]))
        gt_img = self.load_img(os.path.join(self.datapath, self.gt_filenames[index]))
        # add -png.name
        ori_pathname = self.ori_filenames[index]


        if self.training:
            w, h = ori_img.size

            # if w < self.crop_w or h < self.crop_h:
            #
            #     # 图片尺寸比预设的裁剪尺寸小，先同步做resize
            #     ori_img = self.transform_img(ori_img)
            #     gt_img = self.transform_img(gt_img)
            #     # to tensor, normalize  --转为tensor归一化
            #     ori_img = self.processed_color(ori_img)
            #
            #     gt_img = np.array(gt_img, dtype='int64')
            #     gt_img = torch.from_numpy(gt_img)
            #     gt_img = torch.squeeze(gt_img).long()
            #     # gt_img = gt_img.squeeze(0)  # (h, w)
            #
            #     # randomly  images
            #     # do_augment = torch.rand(1).numpy()[0]
            #     # if do_augment > 0.5:
            #     #     ori_img = self.augment_image_pair(ori_img)
            #
            #     return {"ori": ori_img,
            #             "gt": gt_img}

            # random crop --同步裁剪
            x1 = random.randint(0, w - self.crop_w)
            y1 = random.randint(0, h - self.crop_h)

            ori_img = ori_img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
            gt_img = gt_img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))

            # to tensor, normalize  --转为tensor
            ori_img = self.processed_color(ori_img)
            # GT转为tensor时不做归一化
            gt_img = np.array(gt_img, dtype='int64')
            gt_img = torch.from_numpy(gt_img)
            #  nllloss need long() target-GT
            gt_img = torch.squeeze(gt_img).long()  # pytorch-0.4.1

            # gt_img = gt_img.squeeze(0)  # (h, w)

            # randomly  images
            # do_augment = torch.rand(1).numpy()[0]
            # if do_augment > 0.5:
            #     ori_img = self.augment_image_pair(ori_img)

            return {"ori": ori_img,
                    "gt": gt_img}
        else:
            # w, h = ori_img.size
            # 测试时所有图片先resize到crop_w和crop_h尺寸大小
            ori_img = self.transform_img(ori_img)
            gt_img = self.transform_img(gt_img)
            ori_img = processed(ori_img)
            gt_img = np.array(gt_img, dtype='int64')
            gt_img = torch.from_numpy(gt_img)
            gt_img = torch.squeeze(gt_img).long()
            return {"ori": ori_img,
                    "gt": gt_img,
                    "img_name": ori_pathname}
