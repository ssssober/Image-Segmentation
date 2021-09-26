from .pngpng_train import PngPngTrain
from .pngpng_test import PngPng
from .lapa_loader import LapaPngPng

__datasets__ = {
    "pngpngtrain": PngPngTrain,
    "pngpngtest": PngPng,
    "lapa": LapaPngPng
}