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
from deepracing_models.data_loading.image_transforms import IdentifyTransform, AddGaussianNoise
from scipy.interpolate import BSpline, make_interp_spline
import os
import json

def sensibleKnots(t, degree):
    numsamples = t.shape[0]
    knots = [ t[int(numsamples/4)], t[int(numsamples/2)], t[int(3*numsamples/4)] ]
    knots = np.r_[(t[0],)*(degree+1),  knots,  (t[-1],)*(degree+1)]
    return knots

class LocalRacelineDataset(Dataset):
    def __init__(self, root_dir : str, raceline_file : str, context_length : int = 5, dt : float = 2.0, dtype=np.float64):
        super(LocalRacelineDataset, self).__init__()
        poses_dir = os.path.join(root_dir, "image_poses")
        key_file = os.path.join(poses_dir, "image_files.txt")
        with open(key_file, "r") as f:
            self.keys = [key.replace("\n","") for key in f.readlines()]

        npz_file = os.path.join(poses_dir, "geometric_data.npz")
        self.image_poses = np.empty((len(self.keys), 4, 4), dtype=dtype)
        self.image_poses[:,-1] = 0.0
        self.image_poses[:,-1,-1] = 1.0
        with open(npz_file, "rb") as f:
            data = np.load(f)
            self.image_poses[:,0:3,3]=data["interpolated_positions"].astype(dtype).copy()
            self.image_poses[:,0:3,0:3]=Rot.from_quat(data["interpolated_quaternions"]).as_matrix().astype(dtype)

        imagedbdir = os.path.join(root_dir, "images", "lmdb")
        self.image_db_wrapper : ImageLMDBWrapper = ImageLMDBWrapper()
        self.image_db_wrapper.readDatabase(imagedbdir, mapsize=int(len(self.keys)*(66*200*3 + 128)))


        with open(raceline_file, "r") as f:
            racelinedict : dict = json.load(f)
        speedfit : list = racelinedict["speeds"]
        rfit : list = racelinedict["r"]
        tfit : list = racelinedict["t"]
        xfit : list = racelinedict["x"]
        yfit : list = racelinedict["y"]
        zfit : list = racelinedict["z"]

        rlfit = np.column_stack([xfit, yfit, zfit]).astype(dtype)
        finalstretch = rlfit[0] - rlfit[-1]
        finalstretchlength = np.linalg.norm(finalstretch, ord=2, axis=0)
        ratio = 0.975
        extrapoint = rlfit[-1] + ratio*finalstretch
        
        v0 = speedfit[-1]
        vf = v0 + ratio*(speedfit[0] - v0)
        deltad = ratio*finalstretchlength
        a = (vf**2 - v0**2)/(2.0*deltad)

        roots = np.roots([0.5*a, v0, -deltad])
        deltat = (roots[roots>0])[0]

        extrapointspeed=vf
        extrapointtime=tfit[-1] + deltat
        extrapointr=rfit[-1] + deltad

        print()
        print(rlfit[-1])
        print(extrapoint)
        print(rlfit[0])
        print()

        print()
        print(speedfit[-1])
        print(extrapointspeed)
        print(speedfit[0])
        print()

        print()
        print(tfit[-1])
        print(extrapointtime)
        print(tfit[0])
        print()

        rfit.append(extrapointr)
        speedfit.append(extrapointspeed)
        tfit.append(extrapointtime)
        xfit.append(extrapoint[0])
        yfit.append(extrapoint[1])
        zfit.append(extrapoint[2])

        self.raceline = np.column_stack([xfit, yfit, zfit]).astype(dtype)

    def __len__(self):
        return self.image_poses.shape[0] - 1 
    def __getitem__(self, input_index):
        return dict()