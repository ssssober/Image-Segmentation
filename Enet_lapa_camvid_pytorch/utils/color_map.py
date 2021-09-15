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




# VOC数据集中对应的标签 voc-32 class
voc_classes = ["Animal", "Archway","Bicyclist","Bridge","Building","Car","CartLuggagePram",
          "Child","Column_Pole", "Fence", "LaneMkgsDriv", "LaneMkgsNonDriv", "Misc_Text",
          "MotorcycleScooter", "OtherMoving", "ParkingBlock", "Pedestrian", "Road", "RoadShoulder",
          "Sidewalk", "SignSymbol", "Sky", "SUVPickupTruck", "TrafficCone", "TrafficLight",
          "Train", "Tree", "Truck_Bus", "Tunnel", "VegetationMisc", "Void", "Wall"]
# 各种标签所对应的颜色
voc_colormap = [[64,128,64],[192,0,128],[0,128,192],[0,128,64],[128,0,0],[64,0,128],
           [64,0,192],[192,128,64],[192,192,128],[64,64,128],[128,0,192],[192,0,64],
           [128,128,64],[192,0,192],[128,64,64],[64,192,128],[64,64,0],[128,64,128],
           [128,128,192],[0,0,192],[192,128,128],[128,128,128],[64,128,192],[0,0,64],
           [0,64,64],[192,64,128],[128,128,0],[192,128,192],[64,0,64],[192,192,0],
           [0,0,0],[64,192,0]]
voc_cm = np.array(voc_colormap).astype("uint8")

# camvid-13 class
# color_encoding = OrderedDict([
#     ('sky', (128, 128, 128)),
#     ('building', (128, 0, 0)),
#     ('pole', (192, 192, 128)),
#     ('road_marking', (255, 69, 0)),
#     ('road', (128, 64, 128)),
#     ('pavement', (60, 40, 222)),
#     ('tree', (128, 128, 0)),
#     ('sign_symbol', (192, 128, 128)),
#     ('fence', (64, 64, 128)),
#     ('car', (64, 0, 128)),
#     ('pedestrian', (64, 64, 0)),
#     ('bicyclist', (0, 128, 192)),
#     ('unlabeled', (0, 0, 0))
# ])