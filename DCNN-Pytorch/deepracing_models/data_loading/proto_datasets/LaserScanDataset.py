
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
from deepracing.backend import MultiAgentLabelLMDBWrapper, ImageLMDBWrapper, LaserScanLMDBWrapper
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

class LaserScanDataset(Dataset):
    def __init__(self, scan_wrapper : LaserScanLMDBWrapper, label_db_wrapper : MultiAgentLabelLMDBWrapper, keys : List[str], context_length : int,  position_indices : np.ndarray, return_other_agents = False):
        super(LaserScanDataset, self).__init__()
        self.scan_wrapper : LaserScanLMDBWrapper = scan_wrapper
        self.label_db_wrapper : MultiAgentLabelLMDBWrapper = label_db_wrapper
        self.context_length = context_length
        self.db_keys = keys
        self.num_scans = len(self.db_keys)
        self.position_indices = position_indices
        self.return_other_agents = return_other_agents
    def __len__(self):
        return self.num_scans - self.context_length - 1
    def __getitem__(self, input_index):
        label_key = self.db_keys[input_index]
        label_key_idx = int(label_key.split("_")[1])
        images_start = label_key_idx - self.context_length + 1
        images_end = label_key_idx + 1
        packetrange = range(images_start, images_end)
        keys = ["laserscan_%d" % (i,) for i in packetrange]
        assert(keys[-1]==label_key)

        label = self.label_db_wrapper.getMultiAgentLabel(keys[-1])
        
        posespb = label.ego_agent_trajectory.poses
        linearvelspb = label.ego_agent_trajectory.linear_velocities
        session_times = np.asarray([p.session_time for p in posespb])
        egopose = np.eye(4,dtype=np.float64)
        egopose[0:3,3] = np.asarray([label.ego_agent_pose.translation.x, label.ego_agent_pose.translation.y, label.ego_agent_pose.translation.z])
        egopose[0:3,0:3] = Rot.from_quat(np.asarray([label.ego_agent_pose.rotation.x, label.ego_agent_pose.rotation.y, label.ego_agent_pose.rotation.z, label.ego_agent_pose.rotation.w]).astype(np.float64)).as_matrix()

        egopositions = np.asarray([ [p.translation.x, p.translation.y, p.translation.z]  for p in posespb  ])
        egovelocities = np.asarray([ [v.vector.x, v.vector.y, v.vector.z]  for v in linearvelspb  ])

        raceline = np.asarray([ [v.vector.x, v.vector.y, v.vector.z  ]  for v in label.raceline ])

        scanspb = [self.scan_wrapper.getLaserScan(key) for key in keys] 
        scans = np.row_stack([s.ranges for s in scanspb])

        rtndict = {"scans": scans, "ego_current_pose": egopose, "session_times": session_times, "ego_positions": egopositions[:,self.position_indices], "ego_velocities": egovelocities[:,self.position_indices], "raceline": raceline[:,self.position_indices]}

        if self.return_other_agents:
            rtn_agent_positions = 500*np.ones([19,raceline_label.shape[0],raceline_label.shape[1]], dtype=np.float64)
            other_agent_positions = MultiAgentLabelLMDBWrapper.positionsFromLabel(label)
            rtn_agent_positions[0:other_agent_positions.shape[0]] = other_agent_positions
            rtndict["other_agent_positions"] =  rtn_agent_positions[:,:,self.position_indices]

        return rtndict