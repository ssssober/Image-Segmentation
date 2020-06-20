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
import cv2

processed = transforms.Compose([
        transforms.ToTensor(),
    ]
)

class PngPng(Dataset):
    def __init__(self, datapath, list_filename, training, crop_h, crop_w, channel):
        self.datapath = datapath

        self.left_filenames, self.right_filenames = self.load_path(list_filename)
        self.training = training
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.channel = channel

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]

        return left_images, right_images

    def load_image(self, filename):
        if self.channel == 3:
            return Image.open(filename).convert('RGB')
        elif self.channel == 1:
            return Image.open(filename).convert('L')

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))

        # add left_png.name
        left_pathname = self.left_filenames[index]

        if self.training:
            return {"ori": left_img,
                    "gt": right_img}
        else:
            w, h = left_img.size
            x1 = (w - self.crop_w)/2
            y1 = (h - self.crop_h)/2

            # randomly png crop
            left_img = left_img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
            right_img = right_img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))

            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"ori": left_img,
                    "gt": right_img,
                    "left_name": left_pathname}