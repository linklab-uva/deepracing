
import scipy
import scipy.linalg as la
import skimage
import PIL
from PIL import Image as PILImage
import TimestampedPacketMotionData_pb2
import PoseSequenceLabel_pb2
import TimestampedImage_pb2
import Vector3dStamped_pb2
import FrameId_pb2
import Pose3d_pb2
import argparse
import os
import google.protobuf.json_format
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import bisect
import scipy.interpolate
import deepracing.pose_utils
from deepracing.protobuf_utils import getAllImageFilePackets, getAllMotionPackets, getAllSequenceLabelPackets, labelPacketToNumpy
import numpy as np
import torch
from torch.utils.data import Dataset
import skimage
import skimage.io
import torchvision.transforms as transforms
from skimage.transform import resize
import time
import shutil
from tqdm import tqdm as tqdm
from deepracing.imutils import resizeImage as resizeImage
from deepracing.imutils import readImage as readImage
from deepracing.backend import MultiAgentLabelLMDBWrapper, ImageLMDBWrapper
import cv2
import random
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import RotationSpline as RotSpline
from Pose3d_pb2 import Pose3d
from typing import List
import torchvision.transforms as T
import torchvision.transforms.functional as F
from deepracing_models.data_loading.image_transforms import IdentifyTransform
import json
import scipy.interpolate
from scipy.interpolate import make_lsq_spline, BSpline

def sensibleKnots(t, degree):
    numsamples = t.shape[0]
    knots = [ t[int(numsamples/4)], t[int(numsamples/2)], t[int(3*numsamples/4)] ]
    knots = np.r_[(t[0],)*(degree+1),  knots,  (t[-1],)*(degree+1)]
    return knots

def LabelPacketSortKey(packet):
    return packet.car_pose.session_time
def pbPoseToTorch(posepb : Pose3d):
    position_pb = posepb.translation
    rotation_pb = posepb.rotation
    pose = torch.eye(4).double()
    pose[0:3,0:3] = torch.from_numpy( Rot.from_quat( np.array([ rotation_pb.x,rotation_pb.y,rotation_pb.z,rotation_pb.w ], dtype=np.float64 ) ).as_matrix() ).double()
    pose[0:3,3] = torch.from_numpy( np.array( [ position_pb.x, position_pb.y, position_pb.z], dtype=np.float64 )  ).double()

class RacelineLabelDataset(Dataset):
    def __init__(self, image_db_wrapper : ImageLMDBWrapper, label_db_wrapper : MultiAgentLabelLMDBWrapper, keyfile : str, context_length : int, image_size : np.ndarray, position_indices : np.ndarray,\
        raceline_json_file : str, raceline_lookahead: float, extra_transforms : list = [], row_crop_ratio : float = 0.25):
        super(RacelineLabelDataset, self).__init__()
        self.image_db_wrapper : ImageLMDBWrapper = image_db_wrapper
        self.label_db_wrapper : MultiAgentLabelLMDBWrapper = label_db_wrapper
        self.image_size = image_size
        self.context_length = context_length
        self.totensor = transforms.ToTensor()
        with open(keyfile,'r') as filehandle:
            keystrings = filehandle.readlines()
            self.db_keys = [keystring.replace('\n','') for keystring in keystrings]
        self.num_images = len(self.db_keys)
        self.length = self.num_images - 2 - context_length
        self.position_indices = position_indices
        self.transforms = [IdentifyTransform()] + extra_transforms
        with open(raceline_json_file,"r") as f:
            rldict = json.load(f)
        self.raceline_global = np.row_stack([np.array(rldict["x"],dtype=np.float64), np.array(rldict["y"],dtype=np.float64), np.array(rldict["z"],dtype=np.float64),  np.ones_like(np.array(rldict["z"],dtype=np.float64))])
        racelinediffs = np.linalg.norm(self.raceline_global[0:3,1:] -  self.raceline_global[0:3,:-1], ord=2, axis=0)
        self.raceline_buffer = int(round(raceline_lookahead/np.mean(racelinediffs)))
        self.raceline_lookahead = raceline_lookahead
        self.row_crop_ratio = row_crop_ratio


    def __len__(self):
        return self.length*len(self.transforms)
    def __getitem__(self, input_index):
        index = int(input_index%self.num_images)
        label_key = self.db_keys[index]
        label_key_idx = int(label_key.split("_")[1])
        images_start = label_key_idx - self.context_length + 1
        images_end = label_key_idx + 1
        packetrange = range(images_start, images_end)
        keys = ["image_%d" % (i,) for i in packetrange]
        assert(keys[-1]==label_key)

        label = self.label_db_wrapper.getMultiAgentLabel(keys[-1])
        assert(keys[-1]+".jpg"==label.image_tag.image_file)
        rtn_session_times = np.array([p.session_time for p in label.ego_agent_trajectory.poses], dtype=np.float64)
        egopose = np.eye(4,dtype=np.float64)
        egopose[0:3,3] = np.array([label.ego_agent_pose.translation.x, label.ego_agent_pose.translation.y, label.ego_agent_pose.translation.z], dtype=np.float64)
        egopose[0:3,0:3] = Rot.from_quat(np.array([label.ego_agent_pose.rotation.x, label.ego_agent_pose.rotation.y, label.ego_agent_pose.rotation.z, label.ego_agent_pose.rotation.w], dtype=np.float64)).as_matrix()
        egoposeinv = np.linalg.inv(egopose)

        raceline_local = np.matmul(egoposeinv, self.raceline_global)[0:3].transpose()
      #  print("raceline_local.shape: %s " % (str(raceline_local.shape),))
        raceline_distances = np.linalg.norm(raceline_local,ord=2,axis=1)
       # print("raceline_distances.shape: %s " % (str(raceline_distances.shape),))
        closestidx = np.argmin(raceline_distances)
        idxsamp = np.arange(closestidx-int(round(self.raceline_buffer/3)), closestidx+self.raceline_buffer+1,step=1, dtype=np.int64)%raceline_local.shape[0]
     #   print("idxsamp.shape: %s " % (str(idxsamp.shape),))
        raceline_close = raceline_local[idxsamp]
    #    print("raceline_close.shape: %s " % (str(raceline_close.shape),))
        raceline_close = raceline_close[raceline_close[:,self.position_indices[0]]>=0.0]
        raceline_close_dists = np.hstack([np.zeros(1, dtype=np.float64), np.cumsum(np.linalg.norm(raceline_close[1:] - raceline_close[:-1], ord=2, axis=1))])
        k=3
        spl : BSpline = make_lsq_spline(raceline_close_dists, raceline_close, sensibleKnots(raceline_close_dists, k), k=k)
        dsamp = np.linspace(0, raceline_close_dists[-1], num = rtn_session_times.shape[0])
        raceline_label = spl(dsamp)

                
        transform = self.transforms[int(input_index/self.num_images)]
        images_pil = [ PILImage.fromarray( self.image_db_wrapper.getImage(key) ) for key in keys ]
        images_pil = [ F.resize( transform( impil.crop([0, int(round(self.row_crop_ratio*impil.height)), impil.width-1, impil.height-1] ) ) , self.image_size, interpolation=PIL.Image.LANCZOS )  for impil in images_pil ]
        images_torch = torch.stack( [ self.totensor(img) for img in images_pil ] )



        return images_torch, raceline_label[:,self.position_indices], dsamp, 0, 0, packetrange[-1]