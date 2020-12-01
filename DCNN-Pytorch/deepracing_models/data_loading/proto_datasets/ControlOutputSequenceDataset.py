
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
import torchvision, torchvision.transforms.functional as F
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
        self.length = len(self.db_keys) - self.context_length - self.sequence_length - 3
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        image_start = int(self.db_keys[index].split("_")[-1])
        image_end = image_start+self.context_length

        label_start = image_end-1
        label_end = label_start+self.sequence_length

        image_keys = ["image_%d" % i for i in range(image_start, image_end)]
        label_keys = ["image_%d" % i for i in range(label_start, label_end)]

        images = torch.stack( [F.to_tensor(self.image_db_wrapper.getImage(k).copy()) for k in image_keys], dim=0 )

        labels_pb = [self.label_db_wrapper.getControlLabel(k) for k in label_keys]
        assert(str(labels_pb[0].image_file).lower()==(label_keys[0]+".jpg").lower())

        
        steering = np.array([lbl.label.steering for lbl in labels_pb])
        throttle = np.array([lbl.label.throttle for lbl in labels_pb]) 
        brake = np.array([lbl.label.brake for lbl in labels_pb])
    
        return {"images": images, "steering": steering, "throttle": throttle, "brake": brake}