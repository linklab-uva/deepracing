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
class ControlOutputDataset(Dataset):
    def __init__(self, image_db_wrapper : deepracing.backend.ImageLMDBWrapper, label_db_wrapper : deepracing.backend.ControlLabelLMDBWrapper, \
        keyfile, image_size = np.array((66,200))):
        super(ControlOutputDataset, self).__init__()
        self.image_db_wrapper = image_db_wrapper
        self.label_db_wrapper = label_db_wrapper
        self.image_size = image_size
        self.totensor = transforms.ToTensor()
        with open(keyfile,'r') as filehandle:
            keystrings = filehandle.readlines()
            self.db_keys = [keystring.replace('\n','') for keystring in keystrings]
        self.length = len(self.db_keys)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        key = self.db_keys[index]
        label_packet = self.label_db_wrapper.getControlLabel(key)
        image_torch = self.totensor(resizeImage(self.image_db_wrapper.getImage(key), self.image_size))
        control_outputs = torch.zeros(2, dtype=image_torch.dtype)
        throttle = label_packet.label.throttle
        brake = label_packet.label.brake
        accel = throttle-brake
        control_outputs[0] = label_packet.label.steering
        control_outputs[1] = accel
        return image_torch, torch.clamp(control_outputs, -1.0, 1.0)