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
import deepracing.backend
def LabelPacketSortKey(packet):
    return packet.car_pose.session_time
class ControlOutputSequenceDataset(Dataset):
    def __init__(self, image_db_wrapper : deepracing.backend.ImageLMDBWrapper, label_db_wrapper : deepracing.backend.ControlLabelLMDBWrapper, \
        keyfile, image_size = np.array((66,200)), context_length = 5, sequence_length = 1):
        super(ControlOutputSequenceDataset, self).__init__()
        self.image_db_wrapper = image_db_wrapper
        self.label_db_wrapper = label_db_wrapper
        self.image_size = image_size
        self.totensor = transforms.ToTensor()
        self.context_length = context_length
        self.sequence_length = sequence_length
        with open(keyfile,'r') as filehandle:
            keystrings = filehandle.readlines()
            self.db_keys = [keystring.replace('\n','') for keystring in keystrings]
        num_labels = self.label_db_wrapper.getNumLabels()
        self.length = len(self.db_keys) - self.context_length - self.sequence_length - 2
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        image_start = index
        image_end = image_start+self.context_length
        labels_start = image_end-1
        labels_end = labels_start+self.sequence_length

        image_keys = self.db_keys[image_start:image_end]
        label_keys = self.db_keys[labels_start:labels_end]


        image_torch = torch.from_numpy(\
            np.array( [ self.totensor( resizeImage(self.image_db_wrapper.getImage(key), self.image_size)  ).numpy() for key in image_keys ] )\
                )

        control_outputs = torch.zeros(self.sequence_length, 2, dtype=image_torch.dtype)
        for i in range(len(label_keys)):
            label_packet = self.label_db_wrapper.getControlLabel(label_keys[i])
            control_outputs[i][0] = label_packet.label.steering
            accel = label_packet.label.throttle - label_packet.label.brake
            control_outputs[i][1] = accel
        return image_torch, torch.clamp(control_outputs, -1.0, 1.0)