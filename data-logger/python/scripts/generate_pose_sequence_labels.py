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
def toHomogenousTransform(position, quat):
    rtn = np.eye(4)
    rtn[0:3,3] = position
    rtn[0:3,0:3] = quaternion.to_rotation_matrix(quat)
    return rtn
def interpolatePositions(p1, t1, p2, t2, t_interp):
    tau = (t_interp - t1)/(t2 - t1)
    rtn = (1-tau)*p1 + tau*p2
    return rtn
def imageDataKey(data):
    return data.timestamp
def udpPacketKey(packet):
    return packet.timestamp
   # return packet.udp_packet.m_header.m_sessionTime
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
motionPacket = TimestampedPacketMotionData_pb2.TimestampedPacketMotionData()
print(motionPacket.udp_packet.m_angularVelocityX)
filepaths = [f for f in os.listdir(motion_data_folder) if os.path.isfile(os.path.join(motion_data_folder, f)) and str.lower(os.path.splitext(f)[1])==".json"]
print(filepaths)
print(len(filepaths))
jsonstrings = [open(os.path.join(motion_data_folder, path)).read() for path in filepaths]
print(jsonstrings[45])
print(len(jsonstrings))

image_tags = []
image_folder = args.image_path
image_filepaths = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f)) and str.lower(os.path.splitext(f)[1])==".json"]
image_jsonstrings = [open(os.path.join(image_folder, path)).read() for path in image_filepaths]
for jsonstring in image_jsonstrings:
    image_data = TimestampedImage_pb2.TimestampedImage()
    google.protobuf.json_format.Parse(jsonstring, image_data)
    image_tags.append(image_data)
motion_packets = []
for jsonstring in jsonstrings:
    data = TimestampedPacketMotionData_pb2.TimestampedPacketMotionData()
    google.protobuf.json_format.Parse(jsonstring, data)
    motion_packets.append(data)
motion_packets = sorted(motion_packets, key=udpPacketKey)
session_times = np.array([packet.udp_packet.m_header.m_sessionTime for packet in motion_packets])
system_times = np.array([float(packet.timestamp)/1000.0 for packet in motion_packets])
poses = [extractPose(packet.udp_packet) for packet in motion_packets]
print(system_times)
print(session_times)

image_tags = sorted(image_tags, key = imageDataKey)
image_timestamps = np.array([float(data.timestamp)/1000.0 for data in image_tags])
image_timestamps = image_timestamps[image_timestamps<np.max(system_times)]
first_image_time = min(image_timestamps)
print(first_image_time)
I = system_times>first_image_time
firstIndex = np.argmax(I)
#print(I)
#print(firstIndex)
motion_packets = motion_packets[firstIndex:]
session_times = session_times[I]
system_times = system_times[I]
print(len(motion_packets))
print(len(session_times))
print(len(system_times))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(system_times, session_times)
print("Linear map from system time to session time: session_time = %f*system_time + %f" %(slope,intercept))
print("Standard error: %f" %(std_err))
print("R^2: %f" %(r_value**2))
fig = plt.figure("System Time vs F1 Session Time")
plt.plot(system_times, session_times, label='udp data times')
plt.plot(system_times, slope*system_times + intercept, label='fitted line')
#plt.plot(image_timestamps, label='image tag times')
fig.legend()
fig = plt.figure("Session times and remapped image system times")
alltimes = np.sort(np.hstack((session_times, slope*image_timestamps + intercept)))
t = np.linspace( 0.0, 1.0 , num=len(alltimes) )
plt.plot( t, alltimes, label='dem timez' )
plt.show()




