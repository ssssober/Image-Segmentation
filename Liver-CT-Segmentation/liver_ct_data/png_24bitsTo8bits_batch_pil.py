# -*- coding: utf-8 -*-
import os
import numpy as np
import sys
from PIL import Image
# convert 24bits.png to 8bits.png
path = "/**path**/"
for filename in os.listdir(path):
    if os.path.splitext(filename)[1] == '.png':
        img = Image.open(path + filename).convert('L')
        filename = filename[:-4]
        img.save(path + filename + '.png')
print('All png have been convert to npy in the folder of {}'.format(path))

























