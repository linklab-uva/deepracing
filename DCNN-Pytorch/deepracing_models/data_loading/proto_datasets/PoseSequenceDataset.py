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
def LabelPacketSortKey(packet):
    return packet.car_pose.session_time
class PoseSequenceDataset(Dataset):
    def __init__(self, image_db_wrapper, label_db_wrapper, context_length, sequence_length, image_size = np.array((66,200))):
        super(PoseSequenceDataset, self).__init__()
        self.image_db_wrapper = image_db_wrapper
        self.label_db_wrapper = label_db_wrapper
        self.image_size = image_size
        self.context_length = context_length
        self.sequence_length = sequence_length
        self.totensor = transforms.ToTensor()
        self.db_keys = self.label_db_wrapper.getKeys()
        self.label_pb_tags = []
        num_labels = self.label_db_wrapper.getNumLabels()
        print("Preloading database labels.")
        for i,key in tqdm(enumerate(self.db_keys), total=len(self.db_keys)):
            print(key)
            self.label_pb_tags.append(self.label_db_wrapper.getPoseSequenceLabel(key))
            if(not (self.label_pb_tags[-1].image_tag.image_file == self.db_keys[i]+".jpg")):
                raise AttributeError("Mismatch between database key: %s and associated image file: %s" %(self.db_keys[i], self.label_pb_tags.image_tag.image_file))
        self.label_pb_tags = sorted(self.label_pb_tags, key=LabelPacketSortKey)
        self.length = len(self.label_pb_tags) - 1 - context_length
    def resetEnvs(self):
        #pass
        self.image_db_wrapper.resetEnv()
        self.label_db_wrapper.resetEnv()
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        images_start = index
        images_end = index + self.context_length

        packetrange = range(images_start, images_end)
        packets = [self.label_pb_tags[i] for i in packetrange]
        label_packet = packets[-1]

        keys = [os.path.splitext(packets[i].image_tag.image_file)[0] for i in range(len(packets))]

        session_times = np.hstack((np.array([packets[i].car_pose.session_time for i in range(len(packets))]), \
                                   np.array([p.session_time for p in label_packet.subsequent_poses[0:self.sequence_length]])))
        positions, quats, linear_velocities, angular_velocities = deepracing.pose_utils.labelPacketToNumpy(label_packet)
       # tick = time.clock()
        images_torch = torch.from_numpy(np.array([self.totensor(resizeImage(self.image_db_wrapper.getImage(keys[i]), self.image_size)).numpy() for i in range(len(keys))])).float()
        #tock = time.clock()
       # print("loaded images in %f seconds." %(tock-tick))
        positions_torch = torch.from_numpy(positions[0:self.sequence_length]).float()
        quats_torch = torch.from_numpy(quats[0:self.sequence_length]).float()
        linear_velocities_torch = torch.from_numpy(linear_velocities[0:self.sequence_length]).float()
        angular_velocities_torch = torch.from_numpy(angular_velocities[0:self.sequence_length]).float()
        session_times_torch = torch.from_numpy(session_times).float()
        
        return images_torch, positions_torch, quats_torch, linear_velocities_torch, angular_velocities_torch, session_times_torch