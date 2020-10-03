
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
    def __init__(self, image_db_wrapper : ImageLMDBWrapper, label_db_wrapper : MultiAgentLabelLMDBWrapper, keyfile : str, context_length : int, image_size : np.ndarray, position_indices : np.ndarray):
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
        self.position_indices = position_indices

    def __len__(self):
        return self.length
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

        raceline = np.array([[vectorpb.x, vectorpb.y, vectorpb.z]  for vectorpb in label.local_raceline]).astype(np.float64)
        racelinedists = np.array(label.raceline_distances).astype(np.float64)
                
        imagesnp = [ resizeImage(self.image_db_wrapper.getImage(key), self.image_size) for key in keys ]
        images_torch = torch.stack( [ self.totensor(img.copy()) for img in imagesnp ] )

        rtn_agent_positions = np.nan*np.ones([19,raceline.shape[0],raceline.shape[1]], dtype=np.float64)
        other_agent_positions = MultiAgentLabelLMDBWrapper.positionsFromLabel(label)
        rtn_agent_positions[0:other_agent_positions.shape[0]] = other_agent_positions

        rtn_session_times = np.zeros([19,raceline.shape[0]], dtype=np.float64)
        other_agent_times = MultiAgentLabelLMDBWrapper.sessionTimesFromlabel(label)
        rtn_session_times[0:other_agent_times.shape[0]] = other_agent_times

        return images_torch, raceline[:,self.position_indices], racelinedists, rtn_agent_positions[:,:,self.position_indices], rtn_session_times, packetrange[-1]