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
from deepracing.pose_utils import getAllImageFilePackets, getAllMotionPackets, getAllSequenceLabelPackets
import numpy as np
import torch
from torch.utils.data import Dataset
import skimage
import skimage.io
def LabelPacketSortKey(packet):
    return packet.car_pose.session_time
class ProtoDirDataset(Dataset):
    def __init__(self, annotation_directory, context_length, sequence_length):
        super(ProtoDirDataset, self).__init__()
        self.annotation_directory=annotation_directory
        self.image_directory = os.path.dirname(annotion_directory)
        self.label_pb_tags = sorted(deepracing.pose_utils.getAllSequenceLabelPackets(annotation_directory))
        self.context_length = context_length
        self.sequence_length = sequence_length
        self.length = len(self.label_pb_tags) - context_length - sequence_length - 1
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        images_start = index
        images_end = index + self.context_length
        image_files = [os.path.join(self.image_directory,self.label_pb_tags[i].image_tag.image_file) for i in range(images_start, images_end)]
        images_np = np.array([skimage.io.imread(fname) for fname in image_files])        