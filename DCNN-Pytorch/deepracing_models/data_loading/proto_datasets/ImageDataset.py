import torch, torch.utils.data
import torchvision, torchvision.transforms, torchvision.transforms.functional as F
from torch.utils.data.dataset import Dataset
from deepracing.backend import ImageLMDBWrapper
import PIL, PIL.Image as PILImage
import os
import json
import random
import numpy as np
from typing import List

def sortkey(key):
    return int(key.split("_")[1])
class ImageDataset(Dataset):
    def __init__(self, image_wrapper : ImageLMDBWrapper, keys : List[str] = None, image_size=[66,200]):
        super(ImageDataset, self).__init__()
        self.image_wrapper : ImageLMDBWrapper = image_wrapper
        self.image_size = image_size
        if keys is None:
            self.db_keys = sorted(self.image_wrapper.getKeys(), key=sortkey)
        else:
            self.db_keys = sorted(keys, key=sortkey)
        self.num_keys = len(self.db_keys)
        with open(os.path.join(self.image_wrapper.getDBPath(), "timestamps.json"), "r") as f:
            self.timestampdict = json.load(f)
        assert(set(self.db_keys).issubset(set(self.timestampdict.keys())))
    def __len__(self):
        return self.num_keys
    def __getitem__(self, input_index):
        key = self.db_keys[input_index]
        session_time = self.timestampdict[key]
        image_pil =  PILImage.fromarray( self.image_wrapper.getImage(key) )
        image_torch =  F.to_tensor(F.resize(image_pil, self.image_size, interpolation=PIL.Image.LANCZOS))
        rtndict = {"session_time": session_time, "image": image_torch}
        return rtndict
    