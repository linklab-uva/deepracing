
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
from .image_transforms import IdentifyTransform
from Pose3d_pb2 import Pose3d
def LabelPacketSortKey(packet):
    return packet.car_pose.session_time
def pbPoseToTorch(posepb : Pose3d):
    position_pb = posepb.translation
    rotation_pb = posepb.rotation
    pose = torch.eye(4).double()
    pose[0:3,0:3] = torch.from_numpy( Rot.from_quat( np.array([ rotation_pb.x,rotation_pb.y,rotation_pb.z,rotation_pb.w ], dtype=np.float64 ) ).as_matrix() ).double()
    pose[0:3,3] = torch.from_numpy( np.array( [ position_pb.x, position_pb.y, position_pb.z], dtype=np.float64 )  ).double()

class MultiAgentDataset(Dataset):
    def __init__(self, image_db_wrapper : ImageLMDBWrapper, label_db_wrapper : MultiAgentLabelLMDBWrapper, keyfile : str, context_length : int, image_size = np.array((66,200))):
        super(MultiAgentDataset, self).__init__()
        self.image_db_wrapper : ImageLMDBWrapper = image_db_wrapper
        self.label_db_wrapper : MultiAgentLabelLMDBWrapper = label_db_wrapper
        self.image_size = image_size
        self.context_length = context_length
        self.totensor = transforms.ToTensor()
        with open(keyfile,'r') as filehandle:
            keystrings = filehandle.readlines()
            self.db_keys = [keystring.replace('\n','') for keystring in keystrings]
        num_labels = self.label_db_wrapper.getNumLabels()
        self.length = len(self.db_keys) - 2 - context_length

    def __len__(self):
        return self.length
    def __getitem__(self, index):
        images_start = index + 1
        images_end = images_start + self.context_length
        labelsrange = range(images_start, images_end)
        keys = [self.db_keys[i] for i in labelsrange]
        label = self.label_db_wrapper.getMultiAgentLabel(keys[-1])
        ego_states = label.ego_vehicle_path.pose_and_velocities
        ego_poses_pb = [pv.pose for pv in ego_states]
        print(len(ego_poses_pb))
        ego_poses = np.zeros((len(ego_poses_pb),4,4), dtype=np.float64)
        ego_poses[:,3,3]=1.0
        print(ego_poses.shape)
        for i in range(ego_poses.shape[0]):
            ego_position_pb = ego_poses_pb[i].translation
            ego_rotation_pb = ego_poses_pb[i].rotation
            ego_poses[i,0:3,0:3] = Rot.from_quat( np.array([ego_rotation_pb.x,ego_rotation_pb.y,ego_rotation_pb.z,ego_rotation_pb.w], dtype=np.float64 ) ).as_matrix()
            ego_poses[i,0:3,3] =  np.array( [ego_position_pb.x, ego_position_pb.y, ego_position_pb.z], dtype=np.float64 )



        session_times = np.array([p.session_time for p in ego_poses_pb ])
                
        imagesnp = [ resizeImage(self.image_db_wrapper.getImage(key), self.image_size) for key in keys ]
        images_torch = torch.stack( [ self.totensor(img.copy()) for img in imagesnp ]).double()
        #images_torch = images_torch.double()
        return images_torch, torch.from_numpy(ego_poses), torch.from_numpy(session_times)#, agent_positions, agent_velocities