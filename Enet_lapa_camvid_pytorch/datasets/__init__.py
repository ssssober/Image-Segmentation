from .pngpng_train import PngPngTrain
from .pngpng_test import PngPng
from .lapa_loader import LapaPngPng
from .camvid_loader import CamvidPngPng

__datasets__ = {
    "pngpngtrain": PngPngTrain,
    "pngpngtest": PngPng,
    "lapa": LapaPngPng,
    "camvid": CamvidPngPng
}