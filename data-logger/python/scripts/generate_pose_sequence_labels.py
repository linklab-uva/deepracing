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
import cv2
def toHomogenousTransform(position, quat):
    rtn = np.eye(4)
    rtn[0:3,3] = position
    rtn[0:3,0:3] = quaternion.to_rotation_matrix(quat)
    return rtn
def interpolatePositions(p1, t1, p2, t2, t_interp):
    tau = (t_interp - t1)/(t2 - t1)
    rtn = (1-tau)*p1 + tau*p2
    return rtn
def packetKey(packet):
    return packet.udp_packet.m_header.m_sessionTime
def extractPose(packet, car_index = 0):
    motion_data = packet.m_carMotionData[car_index]
   # print(motion_data)
    rightvector = np.array((motion_data.m_worldRightDirX, motion_data.m_worldRightDirY, motion_data.m_worldRightDirZ), dtype=np.float64)
   # rightvector = rightvector/32767.0
    rightvector = rightvector/la.norm(rightvector)
    forwardvector = np.array((motion_data.m_worldForwardDirX, motion_data.m_worldForwardDirY, motion_data.m_worldForwardDirZ), dtype=np.float64)
  #  forwardvector = forwardvector/32767.0
    forwardvector = forwardvector/la.norm(forwardvector)
    upvector = np.cross(rightvector,forwardvector)
    upvector = upvector/la.norm(upvector)
	#rotationMat.col(0) = -right;
	#rotationMat.col(1) = up;
	#rotationMat.col(2) = forward;
    rotationmat = np.vstack((-rightvector,upvector,forwardvector)).transpose()
    #print(rotationmat)
    position = np.array((motion_data.m_worldPositionX, motion_data.m_worldPositionY, motion_data.m_worldPositionZ), dtype=np.float64)
    quat = quaternion.from_rotation_matrix(rotationmat)
    return position, quat 

parser = argparse.ArgumentParser()
parser.add_argument("motion_data_path", help="Path to motion_data packet folder",  type=str)
parser.add_argument("image_path", help="Path to image folder",  type=str)
args = parser.parse_args()
motion_data_folder = args.motion_data_path
image_folder = args.image_path
motionPacket = TimestampedPacketMotionData_pb2.TimestampedPacketMotionData()
print(motionPacket.udp_packet.m_angularVelocityX)
filepaths = [f for f in os.listdir(motion_data_folder) if os.path.isfile(os.path.join(motion_data_folder, f)) and str.lower(os.path.splitext(f)[1])==".json"]
print(filepaths)
print(len(filepaths))
jsonstrings = [open(os.path.join(motion_data_folder, path)).read() for path in filepaths]
print(jsonstrings[45])
print(len(jsonstrings))
data = TimestampedPacketMotionData_pb2.TimestampedPacketMotionData()
motion_packets = [google.protobuf.json_format.Parse(jsonstring,data) for jsonstring in jsonstrings]
motion_packets = sorted(motion_packets, key=packetKey)
session_times = [packet.udp_packet.m_header.m_sessionTime for packet in motion_packets]
system_times = [float(packet.timestamp)/1000.0 for packet in motion_packets]
poses = [extractPose(packet.udp_packet) for packet in motion_packets]
print(motion_packets[45].udp_packet)
print(poses[45])
print(session_times[0])
fig = plt.figure("System Time vs F1 Session Time")
plt.plot(session_times, system_times, label='data')
fig.legend()
plt.show()
