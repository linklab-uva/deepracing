import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import argparse
import numpy as np
import numpy.linalg as la
import scipy
import skimage
import PIL
from PIL import Image as PILImage
import TimestampedPacketMotionData_pb2
import PoseSequenceLabel_pb2
import TimestampedImage_pb2
import Vector3dStamped_pb2
import argparse
import os
import google.protobuf.json_format
import Pose3d_pb2
import cv2
import bisect
import FrameId_pb2
import scipy.interpolate
import deepracing.backend
import deepracing.pose_utils
from deepracing.protobuf_utils import getAllSessionPackets, getAllImageFilePackets, getAllMotionPackets, extractPose, extractVelocity
from tqdm import tqdm as tqdm
import yaml
import shutil
import Spline2DParams_pb2
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import RotationSpline as RotSpline
from scipy.spatial.transform import Slerp
from argparse import ArgumentParser
from tqdm import tqdm as tqdm

parser : ArgumentParser = argparse.ArgumentParser()
parser.add_argument("dataset_dir", help="Path to the directory of the dataset",  type=str)
parser.add_argument("--index",  type=int, default=0, help="Index in the array to get")
args_namespace = parser.parse_args()
args_dict = vars(args_namespace)
print(args_dict)
dataset_dir = args_dict["dataset_dir"]
index = args_dict["index"]

dset_config_file = os.path.join(dataset_dir,"f1_dataset_config.yaml")
with open(dset_config_file,"r") as f:
    dset_config = yaml.load(f,Loader=yaml.SafeLoader)

use_json = dset_config["use_json"]

udp_folder = os.path.join(dataset_dir,"udp_data")
motion_data_folder = os.path.join(udp_folder,"motion_packets")

motion_packets = getAllMotionPackets(motion_data_folder, use_json)
motion_packets = sorted(motion_packets, key=deepracing.timestampedUdpPacketKey)
player_car_indices = [p.udp_packet.m_header.m_playerCarIndex for p in motion_packets]
print(player_car_indices)
player_car_indices_set = set(player_car_indices)
print(player_car_indices_set)
poses = [extractPose(packet.udp_packet, car_index=index) for packet in motion_packets]
velocities = np.array([extractVelocity(packet.udp_packet, car_index=index) for packet in motion_packets])
positions = np.array([pose[0] for pose in poses])
X = positions.copy()
x = X[:,0]
y = X[:,1]
z = X[:,2]
position_diffs = np.diff(positions, axis=0)
position_diff_norms = la.norm(position_diffs, axis=1)
quaternions = np.array([pose[1] for pose in poses])

fig = plt.figure()
#plt.plot(x,z,c="b")
plt.scatter( x, z, marker="o", c="b", s = 0.1*np.ones_like(x) )
plt.savefig(os.path.join(dataset_dir,"udp_data"))
plt.show()