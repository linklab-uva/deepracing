import numpy as np
import numpy.linalg as la
import quaternion
import scipy
import skimage
import PIL
from PIL import Image as PILImage
import TimestampedPacketMotionData_pb2
import TimestampedImageWithPose_pb2
import PoseSequenceLabel_pb2
import TimestampedImage_pb2
import Vector3dStamped_pb2
import argparse
import os
import google.protobuf.json_format
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Pose3d_pb2
import cv2
import bisect
import FrameId_pb2
import scipy.interpolate
import deepracing.pose_utils
from deepracing.pose_utils import getAllImageFilePackets, getAllMotionPackets

def imageDataKey(data):
    return data.timestamp
def udpPacketKey(packet):
    return packet.timestamp

parser = argparse.ArgumentParser()
parser.add_argument("motion_data_path", help="Path to motion_data packet folder",  type=str)
parser.add_argument("image_path", help="Path to image folder",  type=str)
angvelhelp = "Use the angular velocities given in the udp packets. THESE ARE ONLY PROVIDED FOR A PLAYER CAR. IF THE " +\
    " DATASET WAS TAKEN ON SPECTATOR MODE, THE ANGULAR VELOCITY VALUES WILL BE GARBAGE."
parser.add_argument("--use_given_angular_velocities", help=angvelhelp, action="store_true")
parser.add_argument("--assume_linear_timescale", help="Assumes the slope between system time and session time is 1.0", action="store_true")
parser.add_argument("--json", help="Assume dataset files are in JSON rather than binary .pb files.",  action="store_true")
args = parser.parse_args()
motion_data_folder = args.motion_data_path
image_folder = args.image_path
image_tags = deepracing.pose_utils.getAllImageFilePackets(args.image_path, args.json)
motion_packets = deepracing.pose_utils.getAllMotionPackets(args.motion_data_path, args.json)
motion_packets = sorted(motion_packets, key=udpPacketKey)
session_times = np.array([packet.udp_packet.m_header.m_sessionTime for packet in motion_packets])
system_times = np.array([packet.timestamp/1000.0 for packet in motion_packets])
print(system_times)
print(session_times)
maxudptime = system_times[-1]
image_tags = [tag for tag in image_tags if tag.timestamp/1000.0<maxudptime]
image_tags = sorted(image_tags, key = imageDataKey)
image_timestamps = np.array([data.timestamp/1000.0 for data in image_tags])


first_image_time = image_timestamps[0]
print(first_image_time)
Imin = system_times>first_image_time
firstIndex = np.argmax(Imin)

motion_packets = motion_packets[firstIndex:]
motion_packets = sorted(motion_packets, key=udpPacketKey)
session_times = np.array([packet.udp_packet.m_header.m_sessionTime for packet in motion_packets])
unique_session_times, unique_session_time_indices = np.unique(session_times, return_index=True)
motion_packets = [motion_packets[i] for i in unique_session_time_indices]
motion_packets = sorted(motion_packets, key=udpPacketKey)
session_times = np.array([packet.udp_packet.m_header.m_sessionTime for packet in motion_packets])
system_times = np.array([packet.timestamp/1000.0 for packet in motion_packets])

print("Range of session times: [%f,%f]" %(session_times[0], session_times[-1]))
print("Range of udp system times: [%f,%f]" %(system_times[0], system_times[-1]))
print("Range of image system times: [%f,%f]" %(image_timestamps[0], image_timestamps[-1]))

poses = [deepracing.pose_utils.extractPose(packet.udp_packet) for packet in motion_packets]
velocities = np.array([deepracing.pose_utils.extractVelocity(packet.udp_packet) for packet in motion_packets])
positions = np.array([pose[0] for pose in poses])
quaternions = np.array([pose[1] for pose in poses])
if args.use_given_angular_velocities:
    angular_velocities = np.array([deepracing.pose_utils.extractAngularVelocity(packet.udp_packet) for packet in motion_packets])
else:
    angular_velocities = quaternion.angular_velocity(quaternions, session_times)

print()
print(angular_velocities[10])
print(len(motion_packets))
print(len(session_times))
print(len(system_times))
print(len(angular_velocities))
print(len(poses))
print(len(positions))
print(len(velocities))
print(len(quaternions))
print()

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(system_times, session_times)
if args.assume_linear_timescale:
    slope=1.0
image_session_timestamps = slope*image_timestamps + intercept
print("Range of image session times before clipping: [%f,%f]" %(image_session_timestamps[0], image_session_timestamps[-1]))

Iclip = (image_session_timestamps>np.min(session_times)) * (image_session_timestamps<np.max(session_times))
image_tags = [image_tags[i] for i in range(len(image_session_timestamps)) if Iclip[i]]
image_session_timestamps = image_session_timestamps[Iclip]
#and image_session_timestamps<np.max(session_timestamps)
print("Range of image session times after clipping: [%f,%f]" %(image_session_timestamps[0], image_session_timestamps[-1]))


position_interpolant = scipy.interpolate.interp1d(session_times, positions , axis=0, kind='cubic')
#velocity_interpolant = position_interpolant.derivative()
velocity_interpolant = scipy.interpolate.interp1d(session_times, velocities, axis=0, kind='cubic')
interpolated_positions = position_interpolant(image_session_timestamps)
interpolated_velocities = velocity_interpolant(image_session_timestamps)
interpolated_quaternions = quaternion.squad(quaternions, session_times, image_session_timestamps)


interpolated_angular_velocities = quaternion.angular_velocity(interpolated_quaternions, image_session_timestamps)
print()
print(len(image_tags))
print(len(image_session_timestamps))
print(len(interpolated_positions))
print(len(interpolated_quaternions))
print(len(interpolated_angular_velocities))
print()
print("Linear map from system time to session time: session_time = %f*system_time + %f" %(slope,intercept))
print("Standard error: %f" %(std_err))
print("R^2: %f" %(r_value**2))
fig = plt.figure("System Time vs F1 Session Time")
plt.plot(system_times, session_times, label='udp data times')
plt.plot(system_times, slope*system_times + intercept, label='fitted line')
#plt.plot(image_timestamps, label='image tag times')
fig.legend()
fig = plt.figure("Image Session Times on Normalized Domain")
t = np.linspace( 0.0, 1.0 , num=len(image_session_timestamps) )
slope_remap, intercept_remap, r_value_remap, p_value_remap, std_err_remap = scipy.stats.linregress(t, image_session_timestamps)
print("Slope of all point session times" %(slope_remap))
print("Standard error remap: %f" %(std_err_remap))
print("R^2 of remap: %f" %(r_value_remap**2))
plt.plot( t, image_session_timestamps, label='dem timez' )
plt.plot( t, t*slope_remap + intercept_remap, label='fitted line' )
plt.show()
#scipy.interpolate.interp1d
label_folder = "pose_labels"
if(not os.path.isdir(os.path.join(image_folder,label_folder))):
    os.makedirs(os.path.join(image_folder,label_folder))
for idx in range(len(image_tags)):
    label_tag = TimestampedImageWithPose_pb2.TimestampedImageWithPose()
    label_tag.timestamped_image.CopyFrom(image_tags[idx])
    label_tag.pose.frame = FrameId_pb2.GLOBAL
    label_tag.linear_velocity.frame = FrameId_pb2.GLOBAL
    label_tag.angular_velocity.frame = FrameId_pb2.GLOBAL

    t_interp = image_session_timestamps[idx]
    label_tag.pose.session_time = t_interp
    label_tag.linear_velocity.session_time = t_interp
    label_tag.angular_velocity.session_time = t_interp

    carposition_global = interpolated_positions[idx]
    #carposition_global = interpolateVectors(pos1,session_times[i-1],pos2,session_times[i], t_interp)
    carvelocity_global = interpolated_velocities[idx]
    #carvelocity_global = interpolateVectors(vel1,session_times[i-1],vel2,session_times[i], t_interp)
    carquat_global = interpolated_quaternions[idx]
    #carquat_global = quaternion.slerp(quat1,quat2,session_times[i-1],session_times[i], t_interp)
    carangvelocity_global = interpolated_angular_velocities[idx]
    #carangvelocity_global = interpolateVectors(angvel1,session_times[i-1],angvel2,session_times[i], t_interp)
        
    label_tag.pose.translation.x = carposition_global[0]
    label_tag.pose.translation.y = carposition_global[1]
    label_tag.pose.translation.z = carposition_global[2]
    label_tag.pose.rotation.x = carquat_global.x
    label_tag.pose.rotation.y = carquat_global.y
    label_tag.pose.rotation.z = carquat_global.z
    label_tag.pose.rotation.w = carquat_global.w

    label_tag.linear_velocity.vector.x = carvelocity_global[0]
    label_tag.linear_velocity.vector.y = carvelocity_global[1]
    label_tag.linear_velocity.vector.z = carvelocity_global[2]
    
    label_tag.angular_velocity.vector.x = carangvelocity_global[0]
    label_tag.angular_velocity.vector.y = carangvelocity_global[1]
    label_tag.angular_velocity.vector.z = carangvelocity_global[2]

    label_tag_JSON = google.protobuf.json_format.MessageToJson(label_tag, including_default_value_fields=True)
    image_file_base = os.path.splitext(os.path.split(label_tag.timestamped_image.image_file)[1])[0]
    label_tag_file_path = os.path.join(image_folder,label_folder,image_file_base + "_pose_label.json")
    print(label_tag_JSON)
    f = open(label_tag_file_path,'w')
    f.write(label_tag_JSON)
    f.close()
    label_tag_file_path = os.path.join(image_folder,label_folder,image_file_base + "_pose_label.pb")
    f = open(label_tag_file_path,'wb')
    f.write(label_tag.SerializeToString())
    f.close()
    #print(carquatinverse)
   # print(carquat)





