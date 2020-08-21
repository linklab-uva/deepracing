
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
        self.num_images = len(self.db_keys)
        self.length = self.num_images - 2 - context_length

    def __len__(self):
        return self.length
    def __getitem__(self, index):
        index = int(input_index%self.num_images)
        label_key = self.db_keys[index]
        label_key_idx = int(label_key.split("_")[1])
        images_start = label_key_idx - self.context_length + 1
        images_end = label_key_idx + 1
        packetrange = range(images_start, images_end)
        #key_indices = np.array([i for i in packetrange], dtype=np.int32)
        keys = ["image_%d" % (i,) for i in packetrange]
        assert(keys[-1]==label_key)

        label = self.label_db_wrapper.getMultiAgentLabel(keys[-1])
        ego_traj_pb = label.ego_agent_trajectory
        session_times = np.array([p.session_time for p in ego_traj_pb.poses ])

        sizes = np.array( [len(ego_traj_pb.poses), len(ego_traj_pb.linear_velocities), len(ego_traj_pb.angular_velocities)] )
        assert(np.all(sizes == sizes[0]))
        num_points = sizes[0]
        #print(len(ego_traj_pb))
        ego_traj_poses = np.tile(np.eye(4, dtype=np.float64),(num_points,1,1))
        ego_traj_linear_vels = np.zeros((num_points,3), dtype=np.float64)
        ego_traj_angular_vels = np.zeros((num_points,3), dtype=np.float64)
        for i in range(ego_traj_poses.shape[0]):

            ego_pose_pb = ego_traj_pb.poses[i]
            ego_position_pb = ego_pose_pb.translation
            ego_rotation_pb = ego_pose_pb.rotation

            ego_linear_vel_pb = ego_traj_pb.linear_velocities[i]
            ego_traj_linear_vels[i] = np.array( [ego_linear_vel_pb.vector.x, ego_linear_vel_pb.vector.y, ego_linear_vel_pb.vector.z], dtype=np.float64 )

            ego_angular_vel_pb = ego_traj_pb.angular_velocities[i]
            ego_traj_angular_vels[i] = np.array( [ego_angular_vel_pb.vector.x, ego_angular_vel_pb.vector.y, ego_angular_vel_pb.vector.z], dtype=np.float64 )


            q = np.array([ego_rotation_pb.x,ego_rotation_pb.y,ego_rotation_pb.z,ego_rotation_pb.w], dtype=np.float64 )
            ego_traj_poses[i,0:3,0:3] = Rot.from_quat( q/np.linalg.norm(q) ).as_matrix()
            ego_traj_poses[i,0:3,3] =  np.array( [ego_position_pb.x, ego_position_pb.y, ego_position_pb.z], dtype=np.float64 )


        
                
        imagesnp = [ resizeImage(self.image_db_wrapper.getImage(key), self.image_size) for key in keys ]
        images_torch = torch.stack( [ self.totensor(img) for img in imagesnp ] )
        #images_torch = images_torch.double()
        return images_torch, torch.as_tensor(packetrange[-1],dtype=torch.int32), session_times, ego_traj_poses, ego_traj_linear_vels, ego_traj_angular_vels