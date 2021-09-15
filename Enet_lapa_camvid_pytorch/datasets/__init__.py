from .lapa_loader import LapaPngPng
from .camvid_loader import CamvidPngPng

__datasets__ = {
    "lapa": LapaPngPng,
    "camvid": CamvidPngPng
}
