# -*- coding: utf-8 -*-
from pathlib import Path
import os
import random


def read_img_path(img_path):
    data_pathes = sorted(Path(img_path).glob('*.jpg'))  # list

    for path in data_pathes:
        color_path = str(path)
        line = color_path.split('/')
        img_name = line[-1][:-4]
        label_path = '/'.join(line[:-2]) + '/labels/' + img_name + '.png'
        item = color_path + ' ' + label_path
        print(item)
        # 写入txt文本 #####
        File = open('Lapa_test.txt', 'a+')
        File.write(item + '\n')
        File.flush()
        # #####

# def make_(list_path):
#     fopen1 = open(list_path)
#     lines1 = fopen1.readlines()
#     b1 = []
#     for line1 in lines1:
#         line1 = line1.replace('\n', '')

if __name__ == '__main__':
    root_path = '/LaPa/test/images/'
    read_img_path(root_path)
