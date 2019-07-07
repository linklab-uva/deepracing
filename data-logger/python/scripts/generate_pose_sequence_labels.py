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
def fromHomogenousTransform(transform):
    pos = transform[0:3,3].copy()
    quat = quaternion.from_rotation_matrix(transform[0:3,0:3])
    return pos,quat
def toHomogenousTransform(position, quat):
    rtn = np.eye(4)
    rtn[0:3,3] = position.copy()
    rtn[0:3,0:3] = quaternion.as_rotation_matrix(quat)
    return rtn
def interpolateVectors(p1, t1, p2, t2, t_interp):
    tau = (t_interp - t1)/(t2 - t1)
    rtn = (1-tau)*p1 + tau*p2
    return rtn
def imageDataKey(data):
    return data.timestamp
def udpPacketKey(packet):
    return packet.timestamp
   # return packet.udp_packet.m_header.m_sessionTime
def extractVelocity(packet, car_index = 0):
    motion_data = packet.m_carMotionData[car_index]
    velocity = np.array((motion_data.m_worldVelocityX, motion_data.m_worldVelocityY, motion_data.m_worldVelocityZ), np.float64)
    return velocity
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
parser.add_argument("num_label_poses", help="Number of poses to attach to each image",  type=int)
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
    try:
        google.protobuf.json_format.Parse(jsonstring, image_data)
        image_tags.append(image_data)
    except:
        continue
motion_packets = []
for jsonstring in jsonstrings:
    data = TimestampedPacketMotionData_pb2.TimestampedPacketMotionData()
    google.protobuf.json_format.Parse(jsonstring, data)
    motion_packets.append(data)
motion_packets = sorted(motion_packets, key=udpPacketKey)
session_times = np.array([packet.udp_packet.m_header.m_sessionTime for packet in motion_packets])
system_times = np.array([float(packet.timestamp)/1000.0 for packet in motion_packets])
print(system_times)
print(session_times)
maxudptime = system_times[-1]
image_tags = [tag for tag in image_tags if float(tag.timestamp)/1000.0<maxudptime]
image_tags = sorted(image_tags, key = imageDataKey)
image_timestamps = np.array([float(data.timestamp)/1000.0 for data in image_tags])
#Imax = image_timestamps<np.max(system_times)
#lastIndex = np.argmin(Imax)
#image_tags = image_tags[:lastIndex]
#image_timestamps = image_timestamps[Imax]


first_image_time = image_timestamps[0]
print(first_image_time)
Imin = system_times>first_image_time
firstIndex = np.argmax(Imin)

motion_packets = motion_packets[firstIndex:]
session_times = np.array([packet.udp_packet.m_header.m_sessionTime for packet in motion_packets])
system_times = np.array([float(packet.timestamp)/1000.0 for packet in motion_packets])
print(len(motion_packets))
print(len(session_times))
print(len(system_times))


print(len(image_tags))
print(len(image_timestamps))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(system_times, session_times)
image_session_timestamps = slope*image_timestamps + intercept
print("Linear map from system time to session time: session_time = %f*system_time + %f" %(slope,intercept))
print("Standard error: %f" %(std_err))
print("R^2: %f" %(r_value**2))
fig = plt.figure("System Time vs F1 Session Time")
plt.plot(system_times, session_times, label='udp data times')
plt.plot(system_times, slope*system_times + intercept, label='fitted line')
#plt.plot(image_timestamps, label='image tag times')
fig.legend()
fig = plt.figure("Session times and remapped image system times")
alltimes = np.sort(np.hstack((session_times, image_session_timestamps)))
t = np.linspace( 0.0, 1.0 , num=len(alltimes) )
plt.plot( t, alltimes, label='dem timez' )
plt.show()
num_label_poses = args.num_label_poses
for imagetag in image_tags:
    label_tag = PoseSequenceLabel_pb2.PoseSequenceLabel()
    label_tag.image_file = imagetag.image_file
    t_interp = slope*float(imagetag.timestamp)/1000.0+intercept
    label_tag.car_pose.session_time = t_interp
    label_tag.car_velocity.session_time = t_interp
    i = bisect.bisect_left(session_times,t_interp)
    if( i==0 or i > (len(session_times)- num_label_poses) ):
        continue
    print("Image session time: %f. Bisector session time: %f. Bisector index: %d" % (t_interp, session_times[i], i))
    pos1, quat1 = extractPose(motion_packets[i-1].udp_packet)
    pos2, quat2 = extractPose(motion_packets[i].udp_packet)
    vel1 = extractVelocity(motion_packets[i-1].udp_packet)
    vel2 = extractVelocity(motion_packets[i].udp_packet)
    carposition_global = interpolateVectors(pos1,session_times[i-1],pos2,session_times[i], t_interp)
    carquat_global = quaternion.slerp(quat1,quat2,session_times[i-1],session_times[i], t_interp)
    label_tag.car_pose.translation.x = carposition_global[0]
    label_tag.car_pose.translation.y = carposition_global[1]
    label_tag.car_pose.translation.z = carposition_global[2]
    label_tag.car_pose.rotation.x = carquat_global.x
    label_tag.car_pose.rotation.y = carquat_global.y
    label_tag.car_pose.rotation.z = carquat_global.z
    label_tag.car_pose.rotation.w = carquat_global.w
    label_tag.car_pose.frame = FrameId_pb2.GLOBAL
    carvelocity_global = interpolateVectors(vel1,session_times[i-1],vel2,session_times[i], t_interp)
    label_tag.car_velocity.frame = FrameId_pb2.GLOBAL
    label_tag.car_velocity.vector.x = carvelocity_global[0]
    label_tag.car_velocity.vector.y = carvelocity_global[1]
    label_tag.car_velocity.vector.z = carvelocity_global[2]
    carpose_global = toHomogenousTransform(carposition_global, carquat_global)
    #yes, I know this is an un-necessary inverse computation. Sue me.
    carposeinverse_global = la.inv(carpose_global).copy()
    print()
    print()
    print(carposition_global)
    print(carpose_global)
    print()
    for j in range(args.num_label_poses):
        # label_tag.
        pose_forward_pb = Pose3d_pb2.Pose3d()
        velocity_forward_pb = Vector3dStamped_pb2.Vector3dStamped()
        velocity_forward_pb.frame = FrameId_pb2.LOCAL
        pose_forward_pb.frame = FrameId_pb2.LOCAL
        packet_forward = motion_packets[i+j].udp_packet
        pos_forward_global, quat_forward_global = extractPose(packet_forward)
        pose_forward_global = toHomogenousTransform(pos_forward_global, quat_forward_global)
        pose_forward_local = np.matmul(carposeinverse_global, pose_forward_global)
        position_forward_local, quat_forward_local = fromHomogenousTransform(pose_forward_local)
        print()
        print(pose_forward_local)
        print(position_forward_local)
        #print(la.norm(pos_forward_global - carpos_global))
       # print(la.norm(position_forward_numpy_local))
        pose_forward_pb.translation.x = position_forward_local[0]
        pose_forward_pb.translation.y = position_forward_local[1]
        pose_forward_pb.translation.z = position_forward_local[2]
        pose_forward_pb.rotation.x = quat_forward_local.x
        pose_forward_pb.rotation.y = quat_forward_local.y
        pose_forward_pb.rotation.z = quat_forward_local.z
        pose_forward_pb.rotation.w = quat_forward_local.w

        velocity_global = extractVelocity(packet_forward)
        velocity_local = np.matmul(carposeinverse_global[0:3,0:3], velocity_global)
        velocity_forward_pb.vector.x = velocity_local[0]
        velocity_forward_pb.vector.y = velocity_local[1]
        velocity_forward_pb.vector.z = velocity_local[2]
        pose_forward_pb.session_time = packet_forward.m_header.m_sessionTime
        velocity_forward_pb.session_time = packet_forward.m_header.m_sessionTime
        label_tag.subsequent_poses.append(pose_forward_pb)
        label_tag.subsequent_velocities.append(velocity_forward_pb)
    print()
    print()
    label_tag_JSON = google.protobuf.json_format.MessageToJson(label_tag, including_default_value_fields=True)
    image_file_base = os.path.splitext(os.path.split(label_tag.image_file)[1])[0]
    label_tag_file_path = os.path.join(image_folder,image_file_base + "_sequence_label.json")
    f = open(label_tag_file_path,'w')
    f.write(label_tag_JSON)
    f.close()
    #print(carquatinverse)
   # print(carquat)





