from models.enet_ori import enet_ori_model
from models.unet_seg import unet_model
from models.mobilenetV2 import mobileV2_model

__models__ = {
    "enet": enet_ori_model,
    "unet": unet_model,
    "mobilev2": mobileV2_model
}
