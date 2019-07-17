import os
import cv2
import numpy as np
import torchvision.transforms as transforms
import torchvision
import PIL
from PIL import Image as PILImage
import h5py
import skimage
import torch
import torch.random
from torch.utils.data.dataset import Dataset
class DeepRacingH5Dataset:
    def __init__(self, h5filepath: str):
        super(DeepRacingH5Dataset, self).__init__()
    def __getitem__(self, index):
        raise( NotImplementedError("Not yet implemented") )
    def __len__(self):
        raise( NotImplementedError("Not yet implemented") )