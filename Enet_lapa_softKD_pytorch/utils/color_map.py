# -*- coding: utf-8 -*-
import numpy as np





'''
lapa:  11 class
label	class
0	background
1	skin
2	left eyebrow
3	right eyebrow
4	left eye
5	right eye
6	nose
7	upper lip
8	inner mouth
9	lower lip
10	hair
label	color
0	[0, 0, 0]
1	[0, 153, 255]
2	[102, 255, 153]
3	[0, 204, 153]
4	[255, 255, 102]
5	[255, 255, 204]
6	[255, 153, 0]
7	[255, 102, 255]
8	[102, 0, 51]
9	[255, 204, 255]
10	[255, 0, 102]

'''


lapa_map = [[0, 0, 0], [0, 153, 255], [102, 255, 153], [0, 204, 153], [255, 255, 102], [255, 255, 204], [255, 153, 0],
            [255, 102, 255], [102, 0, 51], [255, 204, 255], [255, 0, 102]]
lapa_cm = np.array(lapa_map).astype("uint8")

# lapa_cm[pred[0].cpu().detach().numpy()]




