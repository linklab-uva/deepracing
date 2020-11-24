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
class ImageSequenceDataset(Dataset):
    def __init__(self, image_db_path : str, sequence_length : int, keys : List[str] = None, image_size=[66,200]):
        super(ImageSequenceDataset, self).__init__()
        self.image_wrapper : ImageLMDBWrapper = ImageLMDBWrapper()
        self.image_wrapper.readDatabase(image_db_path)
        self.sequence_length = sequence_length
        self.image_size = image_size
        if keys is None:
            self.db_keys = sorted(self.image_wrapper.getKeys(), key=sortkey)
        else:
            self.db_keys = sorted(keys, key=sortkey)
        self.num_keys = len(self.db_keys)
        with open(os.path.join(image_db_path, "timestamps.json"), "r") as f:
            self.timestampdict = json.load(f)
        assert(set(self.db_keys).issubset(set(self.timestampdict.keys())))
    def __len__(self):
        return self.num_keys - self.sequence_length - 1
    def __getitem__(self, input_index):
        initial_key = self.db_keys[input_index]
        images_start = int(initial_key.split("_")[1])
        images_end = images_start + self.sequence_length
        packetrange = range(images_start, images_end)
        keys = ["image_%d" % (i,) for i in packetrange]
        session_times = np.array([self.timestampdict[key] for key in keys], dtype=np.float64)
        assert(keys[0]==initial_key)        
        images_pil = [ F.resize( PILImage.fromarray( self.image_wrapper.getImage(key) ), self.image_size, interpolation=PIL.Image.LANCZOS ) for key in keys ]
        images_torch = torch.stack( [ F.to_tensor(img) for img in images_pil ] ).double()
        rtndict = {"session_times": session_times, "images": images_torch}
        return rtndict
        
    def statistics(self, sample_size):
        random_keys = random.choices(self.db_keys, k=sample_size)
        images_pil = [ F.resize( PILImage.fromarray( self.image_wrapper.getImage(key) ), self.image_size, interpolation=PIL.Image.LANCZOS ) for key in random_keys ]
        images_torch = torch.stack( [ F.to_tensor(img) for img in images_pil ] )
        stdevs, means = torch.std_mean(images_torch, dim=(0,2,3))
        return means, stdevs