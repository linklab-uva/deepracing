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
from scipy.spatial.kdtree import KDTree

def sensibleKnots(t, degree):
    numsamples = t.shape[0]
    knots = [ t[int(numsamples/4)], t[int(numsamples/2)], t[int(3*numsamples/4)] ]
    knots = np.r_[(t[0],)*(degree+1),  knots,  (t[-1],)*(degree+1)]
    return knots

class LocalRacelineDataset(Dataset):
    def __init__(self, root_dir : str, raceline_file : str, context_length : int = 5, lookahead_time : float = 2.0, dtype=np.float32):
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
        self.inv_image_poses = np.linalg.inv(self.image_poses)

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

        self.tfit : np.ndarray = np.asarray(tfit, dtype=dtype)
        self.rfit : np.ndarray = np.asarray(rfit, dtype=dtype)
        self.speedfit : np.ndarray = np.asarray(speedfit, dtype=dtype)
        self.posfit = np.column_stack([xfit, yfit, zfit]).astype(dtype)
        self.rlspline : BSpline = make_interp_spline(self.tfit, self.posfit) 

        dt = 0.01
        self.tsamp : np.ndarray = np.arange(self.tfit[0], self.tfit[-1], step=dt, dtype=dtype)
        self.possamp : np.ndarray = self.rlspline(self.tsamp).astype(dtype)
        print(self.tsamp.shape)
        self.kdtree : KDTree = KDTree(self.possamp)

        self.lookahead_time = lookahead_time
        self.context_length = context_length

    def __len__(self):
        return self.image_poses.shape[0] - 1 
    def __getitem__(self, i):
        key_idx = int(str.split(self.keys[i], "_")[1])
        keys = ["image_%d"%(j,) for j in range(key_idx-self.context_length+1, key_idx+1)]
        print(key_idx)
        print(keys)
        image_pose = self.image_poses[i]
        image_pose_inv = self.inv_image_poses[i]
        _, iclosest = self.kdtree.query(image_pose[0:3,3])
        t0 = self.tsamp[iclosest]
        tf = t0+self.lookahead_time
        trtn = np.linspace(t0, tf, num=160, dtype=self.tfit.dtype)


        pglobal = self.rlspline(trtn%self.tfit[-1]).astype(self.tfit.dtype)
        pglobal_aug = np.column_stack([pglobal, np.ones_like(pglobal[:,0])])
        plocal = np.matmul(pglobal_aug, image_pose_inv.transpose())[:,0:3]

        pil_images = [F.to_pil_image(self.image_db_wrapper.getImage(key)[1].copy()) for key in keys]
        images = np.stack( [ F.to_tensor(img).numpy().astype(self.tfit.dtype) for img in pil_images ], axis=0 )
        return {"pose": image_pose, "images": images, "t" : trtn, "raceline_positions" : plocal}