import numpy as np
import torchvision.transforms as transforms
import torchvision
import h5py
import torch
from torch.utils.data.dataset import Dataset
import os
class DeepRacingH5Dataset:
    def __init__(self, h5filepath: str):
        super(DeepRacingH5Dataset, self).__init__()
        self.h5file = h5py.File(h5filepath, mode="r")
        self.image_dset = self.h5file["/images"]
        self.position_dset = self.h5file["/position"]
        self.rotation_dset = self.h5file["/rotation"]
        self.linear_velocity_dset = self.h5file["/linear_velocity"]
        self.angular_velocity_dset = self.h5file["/angular_velocity"]
        self.session_time_dset = self.h5file["/session_time"]
        self.imtransform = transforms.ToTensor()
    def __getitem__(self, index):
        image_np = self.image_dset[index]
        position_np = self.position_dset[index]
        rotation_np = self.rotation_dset[index]
        linear_velocity_np = self.linear_velocity_dset[index]
        angular_velocity_np = self.angular_velocity_dset[index]
        session_time = self.session_time_dset[index]
        image_torch = self.imtransform(image_np)
        position_torch = torch.from_numpy(position_np)
        rotation_torch = torch.from_numpy(rotation_np)
        linear_velocity_torch = torch.from_numpy(linear_velocity_np)
        angular_velocity_torch = torch.from_numpy(angular_velocity_np)
        return image_torch, position_torch, rotation_torch, linear_velocity_torch, angular_velocity_torch, session_time
    def __len__(self):
        return self.image_dset.shape[0]