import numpy as np
import numpy.linalg as la
import quaternion
import scipy
import skimage
import PIL
from PIL import Image as PILImage
import TimestampedPacketMotionData_pb2
import argparse
import os
import google.protobuf.json_format
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import TimestampedImage_pb2
import Pose3d_pb2
import cv2
import PoseSequenceLabel_pb2
import bisect
import FrameId_pb2
import Vector3dStamped_pb2
import scipy.interpolate
import deepracing.protobuf_utils
import deepracing.pose_utils as pose_utils
def sortKey(packet):
    return packet.udp_packet.m_header.m_sessionTime
parser = argparse.ArgumentParser()
parser.add_argument("motion_data_dir", help="Path to motion data to generate trackfile from",  type=str)
parser.add_argument("--trackfileout", help="Path to an ARMA format matrix file",  type=str, default="track.arma")
parser.add_argument("--json", action="store_true", help="Look for json files in motion_data_dir instead of binary .pb files")

args = parser.parse_args()
motion_data_dir = args.motion_data_dir
use_json = args.json 
trackfileout = args.trackfileout
motion_packets = sorted(pose_utils.getAllMotionPackets(motion_data_dir, args.json), key=sortKey)
#print(motion_packets)

car_index = 0
poses = [ pose_utils.extractPose(p.udp_packet, car_index = car_index) for p in motion_packets]
t =  np.array([sortKey(p) for p in motion_packets])
X = np.array([ pose[0] for pose in poses])
Xdot = np.array([pose_utils.extractVelocity(p.udp_packet, car_index = car_index) for p in motion_packets])
_,unique_indices = np.unique(t,return_index=True)
t = t[unique_indices]
X = X[unique_indices]
Xdot = Xdot[unique_indices]

Xdotnorm = Xdot.copy()
for i in range(Xdotnorm.shape[0]):
    Xdotnorm[i,:] = Xdotnorm[i,:]/la.norm(Xdotnorm[i,:])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c='r', marker='o', s = np.ones_like(X[:,0]))
ax.quiver(X[:,0], X[:,1], X[:,2], Xdotnorm[:,0], Xdotnorm[:,1], Xdotnorm[:,2])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

if not os.path.isdir(os.path.dirname(trackfileout)):
    os.makedirs(os.path.dirname(trackfileout))
deepracing.writeArmaFile(trackfileout,t,X,Xdot)