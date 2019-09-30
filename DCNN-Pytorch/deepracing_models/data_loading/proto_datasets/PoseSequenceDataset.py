import quaternion
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
from deepracing.pose_utils import getAllImageFilePackets, getAllMotionPackets, getAllSequenceLabelPackets, labelPacketToNumpy
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
import cv2
import random
from .image_transforms import IdentifyTransform
def LabelPacketSortKey(packet):
    return packet.car_pose.session_time
class PoseSequenceDataset(Dataset):
    def __init__(self, image_db_wrapper, label_db_wrapper, keyfile, context_length, image_size = np.array((66,200)), optical_flow_db_wrapper=None, erasing_probability=0.0, apply_color_jitter = False):
        super(PoseSequenceDataset, self).__init__()
        self.image_db_wrapper = image_db_wrapper
        self.label_db_wrapper = label_db_wrapper
        self.optical_flow_db_wrapper=optical_flow_db_wrapper
        self.image_size = image_size
        self.context_length = context_length
        self.totensor = transforms.ToTensor()
        self.topil = transforms.ToPILImage()
        if erasing_probability>0.0:
            self.erasing = transforms.RandomErasing(p=erasing_probability)
        else:
            self.erasing = IdentifyTransform()
        if apply_color_jitter:
            self.colorjitter = transforms.ColorJitter(brightness=(0.75,1.5), contrast=0.2)
        else:
            self.colorjitter = IdentifyTransform()
        with open(keyfile,'r') as filehandle:
            keystrings = filehandle.readlines()
            self.db_keys = [keystring.replace('\n','') for keystring in keystrings]
        num_labels = self.label_db_wrapper.getNumLabels()
        # print("Preloading database labels.")
        # for i,key in tqdm(enumerate(self.db_keys), total=len(self.db_keys)):
        #     #print(key)
        #     self.label_pb_tags.append(self.label_db_wrapper.getPoseSequenceLabel(key))
        #     if(not (self.label_pb_tags[-1].image_tag.image_file == self.db_keys[i]+".jpg")):
        #         raise AttributeError("Mismatch between database key: %s and associated image file: %s" %(self.db_keys[i], self.label_pb_tags.image_tag.image_file))
        # self.label_pb_tags = sorted(self.label_pb_tags, key=LabelPacketSortKey)
        self.length = len(self.db_keys) - 5 - context_length
    def resetEnvs(self):
        #pass
        self.image_db_wrapper.resetEnv()
        self.label_db_wrapper.resetEnv()
    def clearReaders(self):
        self.image_db_wrapper.clearStaleReaders()
        self.label_db_wrapper.clearStaleReaders()
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        images_start = index + 1
        images_end = images_start + self.context_length
        packetrange = range(images_start, images_end)
        keys = [self.db_keys[i] for i in packetrange]
        packets = [self.label_db_wrapper.getPoseSequenceLabel(keys[i]) for i in range(len(keys))]
        label_packet = packets[-1]


        session_times = np.array([p.session_time for p in label_packet.subsequent_poses ])
        positions, quats, linear_velocities, angular_velocities = deepracing.pose_utils.labelPacketToNumpy(label_packet)
        
        
        positions_np = positions
        linear_velocities_np = linear_velocities
        angular_velocities_np = angular_velocities
        quats_np = quats
        session_times_np = session_times
            
      
       # tick = time.clock()
        #tock = time.clock()
       # print("loaded images in %f seconds." %(tock-tick))
        positions_torch = torch.from_numpy(positions_np)
        quats_torch = torch.from_numpy(quats_np)
        linear_velocities_torch = torch.from_numpy(linear_velocities_np)
        angular_velocities_torch = torch.from_numpy(angular_velocities_np)
        session_times_torch = torch.from_numpy(session_times_np)
        #pos_spline_params = torch.from_numpy(np.vstack((np.array(label_packet.position_spline.XParams),np.array(label_packet.position_spline.ZParams))))
       # vel_spline_params = torch.from_numpy(np.vstack((np.array(label_packet.velocity_spline.XParams),np.array(label_packet.velocity_spline.ZParams))))
       # knots = torch.from_numpy(np.array(label_packet.position_spline.knots))
        imagesnp = [resizeImage(self.image_db_wrapper.getImage(keys[i]), self.image_size) for i in range(len(keys))]
        if random.choice([True,False]):
            pilimages = [transforms.functional.hflip(self.topil(img)) for img in imagesnp]
            positions_torch[:,0]*=-1.0
            linear_velocities_torch[:,0]*=-1.0
            angular_velocities_torch[:,[1,2]]*=-1.0
        else:
            pilimages = [self.topil(img) for img in imagesnp]
        if (not isinstance(self.colorjitter,IdentifyTransform)) and random.choice([True,False]):
            pilimages = [self.colorjitter(img) for img in pilimages]
       # pilimages = [self.erasing(img) for img in pilimages]    
        images_torch = torch.stack([self.erasing(self.totensor(img)) for img in pilimages])
        if self.optical_flow_db_wrapper is not None:
            opt_flow_torch = torch.from_numpy( np.array([self.optical_flow_db_wrapper.getImage(keys[i]).transpose(2,0,1) for i in range(len(keys))]) ).type_as(images_torch)
        else:
            opt_flow_torch = torch.zeros_like(images_torch)
            opt_flow_torch.fill_(np.nan)
        return images_torch, opt_flow_torch, positions_torch, quats_torch, linear_velocities_torch, angular_velocities_torch, session_times_torch#, pos_spline_params, vel_spline_params, knots