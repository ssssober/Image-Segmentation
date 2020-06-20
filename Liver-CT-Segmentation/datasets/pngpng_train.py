# -*- coding: utf-8 -*-
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines
import torchvision.transforms as transforms
import torch
import torch.nn as nn

processed = transforms.Compose([transforms.ToTensor(),])

class PngPngTrain(Dataset):
    def __init__(self, datapath, list_filename, training, crop_h, crop_w, channel):
        self.datapath = datapath
        self.ori_filenames, self.gt_filenames = self.load_path(list_filename)
        self.training = training
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.channel = channel

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        ori_images = [x[0] for x in splits]
        gt_images = [x[1] for x in splits]
        return ori_images, gt_images

    def load_image(self, filename):
        if self.channel == 3:
            return Image.open(filename).convert('RGB')
        elif self.channel == 1:
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
        gt_img = self.load_image(os.path.join(self.datapath, self.gt_filenames[index]))

        if self.training:
            w, h = ori_img.size

            x1 = random.randint(0, w - self.crop_w)
            y1 = random.randint(0, h - self.crop_h)

            # random crop
            ori_img = ori_img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
            gt_img = gt_img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))

            # to tensor, normalize
            ori_img = processed(ori_img)
            gt_img = processed(gt_img)

            # randomly  images
            do_augment = torch.rand(1).numpy()[0]
            if do_augment > 0.5:
                ori_img = self.augment_image_pair(ori_img)
            # print(ori_img.size())
            return {"ori": ori_img,
                    "gt": gt_img}
        else:
            w, h = ori_img.size
            # crop_w, crop_h = 960, 512
            x1 = w - self.crop_w
            y1 = h - self.crop_h

            ori_img = ori_img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
            gt_img = gt_img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))

            ori_img = processed(ori_img)
            gt_img = processed(gt_img)
            return {"ori": ori_img,
                    "gt": gt_img}
