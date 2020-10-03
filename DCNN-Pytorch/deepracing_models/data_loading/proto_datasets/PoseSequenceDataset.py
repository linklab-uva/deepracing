
import scipy
import scipy.linalg as la
import skimage
import PIL
from PIL import Image as PILImage
from PIL.ImageFilter import GaussianBlur
import PIL.ImageFilter
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
import torchvision.transforms.functional as F
from skimage.transform import resize
import time
import shutil
from tqdm import tqdm as tqdm
from deepracing.imutils import resizeImage as resizeImage
from deepracing.imutils import readImage as readImage
import cv2
import random
from itertools import chain, combinations
from scipy.spatial.transform import Rotation as Rot

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return [list(elem) for elem in list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1))) if elem!=tuple([]) ]


def LabelPacketSortKey(packet):
    return packet.car_pose.session_time
class PoseSequenceDataset(Dataset):
    def __init__(self, image_db_wrapper, label_db_wrapper, keyfile, context_length, track_id,\
        image_size = np.array((66,200)), lookahead_indices = -1, lateral_dimension = 1,\
            gaussian_blur = None, color_jitter = None, geometric_variants = False):
        super(PoseSequenceDataset, self).__init__()
        self.image_db_wrapper = image_db_wrapper
        self.label_db_wrapper = label_db_wrapper
        self.image_size = image_size
        self.context_length = context_length
        self.totensor = transforms.ToTensor()
        self.topil = transforms.ToPILImage()
        self.lateral_dimension = lateral_dimension
        self.lookahead_indices = lookahead_indices
        self.track_id = track_id
        if bool(gaussian_blur) and gaussian_blur>0:
            self.gaussian_blur_PIL = GaussianBlur(radius=gaussian_blur)
            self.gaussian_blur = transforms.Lambda(lambda img : img.filter(self.gaussian_blur_PIL))
        else:
            self.gaussian_blur = None
        if bool(color_jitter):
            self.color_jitter = transforms.ColorJitter( brightness=(color_jitter[0],color_jitter[0]), contrast=(color_jitter[1],color_jitter[1]) )
        else:
            self.color_jitter = None
        if geometric_variants:
            self.hflip = transforms.Lambda(lambda img : transforms.functional.hflip(img))
        else:
            self.hflip = None
        with open(keyfile,'r') as filehandle:
            keystrings = filehandle.readlines()
            self.db_keys = [keystring.replace('\n','') for keystring in keystrings]
        
        self.num_images = len(self.db_keys)# - 5 - context_length
        self.transform_dict = {0: lambda img : img}
        overflowidx = 1
        possible_transforms = [ tf for tf in [self.gaussian_blur, self.color_jitter , self.hflip ] if bool(tf) ]
        self.transform_powerset = powerset(possible_transforms)
        self.transform_powerset.append([transforms.Lambda(lambda img : img)])
        self.length = self.num_images*len(self.transform_powerset)
        # if bool(self.gaussian_blur):
        #     self.transform_dict[overflowidx] = 
        #     overflowidx+=1
        #     self.length += self.num_images
        # if bool(self.color_jitter):
        #     self.transform_dict[overflowidx] = self.color_jitter
        #     overflowidx+=1
        #     self.length += self.num_images
        # if geometric_variants:
        #     self.transform_dict[overflowidx] = "flip_images"
        #     overflowidx+=1
        #     self.length += self.num_images

    def resetEnvs(self):
        #pass
        self.image_db_wrapper.resetEnv()
        self.label_db_wrapper.resetEnv()
    def clearReaders(self):
        self.image_db_wrapper.clearStaleReaders()
        self.label_db_wrapper.clearStaleReaders()
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

       # packets = [self.label_db_wrapper.getPoseSequenceLabel(keys[i]) for i in range(len(keys))]
        label_packet = self.label_db_wrapper.getPoseSequenceLabel(keys[-1])


        session_times_np = np.array([p.session_time for p in label_packet.subsequent_poses ])
        positions_np, quats_np, linear_velocities_np, angular_velocities_np = deepracing.protobuf_utils.labelPacketToNumpy(label_packet)
        car_pose_proto = label_packet.car_pose
        car_pose_np = np.eye(4, dtype=np.float64)
        car_pose_np[0:3,3] =  np.array([car_pose_proto.translation.x, car_pose_proto.translation.y, car_pose_proto.translation.z], dtype=np.float64)
        car_pose_np[0:3,0:3] = Rot.from_quat(np.array([car_pose_proto.rotation.x, car_pose_proto.rotation.y, car_pose_proto.rotation.z, car_pose_proto.rotation.w], dtype=np.float64)).as_matrix()
        
        positions_torch = torch.from_numpy(positions_np).double()
        quats_torch = torch.from_numpy(quats_np).double()
        linear_velocities_torch = torch.from_numpy(linear_velocities_np).double()
        angular_velocities_torch = torch.from_numpy(angular_velocities_np).double()
        session_times_torch = torch.from_numpy(session_times_np).double()

        max_i = min(self.lookahead_indices, positions_torch.shape[0])
        if max_i>0:
            positions_torch = positions_torch[0:max_i]
            quats_torch = quats_torch[0:max_i]
            linear_velocities_torch = linear_velocities_torch[0:max_i]
            angular_velocities_torch = angular_velocities_torch[0:max_i]
            session_times_torch = session_times_torch[0:max_i]

        pilimages = [ self.topil(resizeImage(self.image_db_wrapper.getImage(keys[i]), self.image_size)) for i in range(len(keys)) ]

        quotient = int(input_index/self.num_images)
        transform_list = self.transform_powerset[quotient]
        transform = transforms.Compose(transform_list)
        pilimages = [transform(img) for img in pilimages] 
        if self.hflip in transform_list:
            positions_torch[:,self.lateral_dimension]*=-1.0
            linear_velocities_torch[:,self.lateral_dimension]*=-1.0
            angular_velocities_torch[:,[i for i in range(3) if i!=self.lateral_dimension]]*=-1.0
        # else:
        #     tf = self.transform_dict[quotient]
        images_torch = torch.stack( [ self.totensor(img) for img in pilimages ] ).double()
       

        return image_keys, images_torch, positions_torch, quats_torch, linear_velocities_torch, angular_velocities_torch, session_times_torch, car_pose_np, self.track_id