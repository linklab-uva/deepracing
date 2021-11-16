import torch
import scipy.linalg as la
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm as tqdm
from deepracing.backend import  ImageLMDBWrapper
from scipy.spatial.transform import Rotation as Rot
import torchvision.transforms as T
import torchvision.transforms.functional as F
from deepracing_models.data_loading.image_transforms import IdentityTransform, AddGaussianNoise
from scipy.interpolate import BSpline, make_interp_spline
import os
import json
from scipy.spatial.kdtree import KDTree

def sensibleKnots(t, degree):
    numsamples = t.shape[0]
    knots = [ t[int(numsamples/4)], t[int(numsamples/2)], t[int(3*numsamples/4)] ]
    knots = np.r_[(t[0],)*(degree+1),  knots,  (t[-1],)*(degree+1)]
    return knots

class FutureEgoPoseDataset(Dataset):
    def __init__(self, root_dir : str, sample_count = 160, context_length : int = 5, lookahead_time : float = 2.0, dtype=np.float32):
        super(FutureEgoPoseDataset, self).__init__()
        poses_dir = os.path.join(root_dir, "image_poses")
        key_file = os.path.join(poses_dir, "image_files.txt")
        with open(key_file, "r") as f:
            self.keys = [key.replace("\n","") for key in f.readlines()]

        npz_file = os.path.join(poses_dir, "geometric_data.npz")
        self.image_poses : np.ndarray = np.empty((len(self.keys), 4, 4), dtype=dtype)
        self.image_poses[:,-1] = 0.0
        self.image_poses[:,-1,-1] = 1.0
        with open(npz_file, "rb") as f:
            data = np.load(f)
            self.image_poses[:,0:3,3] = data["interpolated_positions"].astype(dtype).copy()
            self.image_poses[:,0:3,0:3] = Rot.from_quat(data["interpolated_quaternions"]).as_matrix().astype(dtype)
            self.image_session_timestamps : np.ndarray = data["image_session_timestamps"].astype(dtype).copy()
            self.tfit : np.ndarray = data["udp_session_times"].astype(dtype).copy()
            self.positionfit : np.ndarray = data["udp_positions"].astype(dtype).copy()
            self.velocityfit : np.ndarray = data["udp_velocities"].astype(dtype).copy()
        idxclip = (self.image_session_timestamps > (np.min(self.tfit) + 1.25*lookahead_time) )*(self.image_session_timestamps < (np.max(self.tfit) - 1.25*lookahead_time) )
        self.image_poses = self.image_poses[idxclip]
        self.image_session_timestamps = self.image_session_timestamps[idxclip]
        self.keys = [self.keys[i] for i in range(len(self.keys)) if idxclip[i]]

        imagedbdir = os.path.join(root_dir, "images", "lmdb")
        self.image_db_wrapper : ImageLMDBWrapper = ImageLMDBWrapper()
        self.image_db_wrapper.readDatabase(imagedbdir, mapsize=int(len(self.keys)*(66*200*3 + 128)))

        self.lookahead_time = lookahead_time
        self.context_length = context_length
        self.sample_count=sample_count

        self.positionspline : BSpline = make_interp_spline(self.tfit, self.positionfit)
        self.velocityspline : BSpline = make_interp_spline(self.tfit, self.velocityfit)

    def __len__(self):
        return self.image_poses.shape[0] - 1 
    def __getitem__(self, i):
        key_idx = int(str.split(self.keys[i], "_")[1])
        keys = ["image_%d"%(j,) for j in range(key_idx-self.context_length+1, key_idx+1)]
        image_pose = self.image_poses[i]
        t0 = self.image_session_timestamps[i]
        tf = t0+self.lookahead_time
        trtn = np.linspace(t0, tf, num=self.sample_count, dtype=self.tfit.dtype)

        pglobal = self.positionspline(trtn%self.tfit[-1]).astype(self.tfit.dtype)
        vglobal = self.velocityspline(trtn%self.tfit[-1]).astype(self.tfit.dtype)

        pil_images = [F.to_pil_image(self.image_db_wrapper.getImage(key)[1].copy()) for key in keys]
        images = np.stack( [ F.to_tensor(img).numpy().astype(self.tfit.dtype) for img in pil_images ], axis=0 )
        
        return {"pose": image_pose, "images": images, "t" : trtn, "positions" : pglobal, "velocities" : vglobal}