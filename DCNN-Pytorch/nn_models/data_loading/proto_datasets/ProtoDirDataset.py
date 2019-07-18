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
def LabelPacketSortKey(packet):
    return packet.car_pose.session_time
class ProtoDirDataset(Dataset):
    def __init__(self, annotation_directory, context_length, sequence_length):
        super(ProtoDirDataset, self).__init__()
        self.annotation_directory=annotation_directory
        self.image_directory = os.path.dirname(annotation_directory)
        self.label_pb_tags = sorted(deepracing.pose_utils.getAllSequenceLabelPackets(annotation_directory, use_json=True), key=LabelPacketSortKey)
        print([tag.image_tag.image_file for tag in self.label_pb_tags])
        self.context_length = context_length
        self.sequence_length = sequence_length
        self.length = len(self.label_pb_tags) - context_length# - sequence_length - 1
        self.totensor = transforms.ToTensor()
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        images_start = index
        images_end = index + self.context_length
        packetrange = range(images_start, images_end)
        image_files = [os.path.join(self.image_directory,self.label_pb_tags[i].image_tag.image_file) for i in packetrange]
        print(image_files)
        image_collection = skimage.io.imread_collection(image_files)
        images = torch.from_numpy(np.array([self.totensor(image_collection[i]).numpy() for i in range(len(image_files))]))
        
        label_packet = self.label_pb_tags[images_end-1]
        session_times = np.hstack((np.array([self.label_pb_tags[i].car_pose.session_time for i in packetrange]), \
                                   np.array([p.session_time for p in label_packet.subsequent_poses[0:self.sequence_length]])))
        positions, quats, linear_velocities, angular_velocities = deepracing.pose_utils.labelPacketToNumpy(label_packet)
        positions_torch = torch.from_numpy(positions[0:self.sequence_length])
        quats_torch = torch.from_numpy(quats[0:self.sequence_length])
        linear_velocities_torch = torch.from_numpy(linear_velocities[0:self.sequence_length])
        angular_velocities_torch = torch.from_numpy(angular_velocities[0:self.sequence_length])
        session_times_torch = torch.from_numpy(session_times)
        
        return images, positions_torch, quats_torch, linear_velocities_torch, angular_velocities_torch, session_times_torch