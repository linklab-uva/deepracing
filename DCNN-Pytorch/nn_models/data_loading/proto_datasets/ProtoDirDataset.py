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
from skimage.transform import rescale, resize, downscale_local_mean
import time
import shutil
from tqdm import tqdm as tqdm
def LabelPacketSortKey(packet):
    return packet.car_pose.session_time
class ProtoDirDataset(Dataset):
    def __init__(self, annotation_directory, context_length, sequence_length, cache_images=True, image_size = np.array((66,200)), image_directory= None):
        super(ProtoDirDataset, self).__init__()
        if annotation_directory.endswith(os.path.sep):
            self.annotation_directory=annotation_directory[0:len(annotation_directory)-1]
        else:
            self.annotation_directory=annotation_directory
        self.image_size=image_size
        self.label_pb_tags = sorted(deepracing.pose_utils.getAllSequenceLabelPackets(self.annotation_directory, use_json=True), key=LabelPacketSortKey)
        print([tag.image_tag.image_file for tag in self.label_pb_tags])
        self.context_length = context_length
        self.sequence_length = sequence_length
        self.length = len(self.label_pb_tags) - context_length# - sequence_length - 1
        self.totensor = transforms.ToTensor()
        if image_directory is None:
            image_directory = os.path.dirname(self.annotation_directory)
            print("Resizing images to temporary directory")
            tmpdir = ("tmpdir_" + time.asctime()).replace(" ","_").replace(":","_")
            os.mkdir(tmpdir)
            for pb_tag in tqdm(self.label_pb_tags):
                img_in = skimage.util.img_as_ubyte(skimage.io.imread(os.path.join(image_directory, pb_tag.image_tag.image_file)))
                img_rs = skimage.util.img_as_ubyte(resize(img_in, image_size))
                skimage.io.imsave(os.path.join(tmpdir, pb_tag.image_tag.image_file), img_rs)
            self.image_files = [os.path.join(tmpdir, pb_tag.image_tag.image_file) for pb_tag in self.label_pb_tags]
        else:
            self.image_files = [os.path.join(image_directory, pb_tag.image_tag.image_file) for pb_tag in self.label_pb_tags]
        
        #print(image_files)
        self.image_collection = skimage.io.imread_collection(self.image_files, conserve_memory=not cache_images)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        images_start = index
        images_end = index + self.context_length
        packetrange = range(images_start, images_end)
        
        label_packet = self.label_pb_tags[images_end-1]
        session_times = np.hstack((np.array([self.label_pb_tags[i].car_pose.session_time for i in packetrange]), \
                                   np.array([p.session_time for p in label_packet.subsequent_poses[0:self.sequence_length]])))
        positions, quats, linear_velocities, angular_velocities = deepracing.pose_utils.labelPacketToNumpy(label_packet)
     #   print("loading images")
        tick = time.clock()
        images_torch = torch.from_numpy(np.array([self.totensor(skimage.util.img_as_ubyte(resize(self.image_collection[i], self.image_size))).numpy() for i in packetrange])).float()
        tock = time.clock()
       # print("loaded images in %f seconds." %(tock-tick))
        positions_torch = torch.from_numpy(positions[0:self.sequence_length]).float()
        quats_torch = torch.from_numpy(quats[0:self.sequence_length]).float()
        linear_velocities_torch = torch.from_numpy(linear_velocities[0:self.sequence_length]).float()
        angular_velocities_torch = torch.from_numpy(angular_velocities[0:self.sequence_length]).float()
        session_times_torch = torch.from_numpy(session_times).float()
        
        return images_torch, positions_torch, quats_torch, linear_velocities_torch, angular_velocities_torch, session_times_torch