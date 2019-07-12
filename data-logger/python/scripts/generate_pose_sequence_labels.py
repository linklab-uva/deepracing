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
def extractAngularVelocity(packet):
    angular_velocity = np.array((packet.m_angularVelocityX, packet.m_angularVelocityY, packet.m_angularVelocityZ), np.float64)
    return angular_velocity
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
angvelhelp = "Use the angular velocities given in the udp packets. THESE ARE ONLY PROVIDED FOR A PLAYER CAR. IF THE " +\
    " DATASET WAS TAKEN ON SPECTATOR MODE, THE ANGULAR VELOCITY VALUES WILL BE GARBAGE."
parser.add_argument("--use_given_angular_velocities", help=angvelhelp, action="store_true")
parser.add_argument("--assume_linear_timescale", help="Assumes the slope between system time and session time is 1.0", action="store_true")
parser.add_argument("--json", help="Assume dataset files are in JSON rather than binary .pb files.",  action="store_true")
args = parser.parse_args()
motion_data_folder = args.motion_data_path
image_folder = args.image_path
image_tags = []
motion_packets = []
if args.json:
    image_filepaths = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f)) and str.lower(os.path.splitext(f)[1])==".json"]
    image_jsonstrings = [open(os.path.join(image_folder, path)).read() for path in image_filepaths]
    for jsonstring in image_jsonstrings:
        image_data = TimestampedImage_pb2.TimestampedImage()
        try:
            google.protobuf.json_format.Parse(jsonstring, image_data)
            image_tags.append(image_data)
        except:
            continue
    filepaths = [f for f in os.listdir(motion_data_folder) if os.path.isfile(os.path.join(motion_data_folder, f)) and str.lower(os.path.splitext(f)[1])==".json"]
    print(filepaths)
    print(len(filepaths))
    jsonstrings = [open(os.path.join(motion_data_folder, path)).read() for path in filepaths]
    print(jsonstrings[45])
    print(len(jsonstrings))
    for jsonstring in jsonstrings:
        data = TimestampedPacketMotionData_pb2.TimestampedPacketMotionData()
        google.protobuf.json_format.Parse(jsonstring, data)
        motion_packets.append(data)
else:
    image_filepaths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f)) and str.lower(os.path.splitext(f)[1])==".pb"]
    for image_filepath in image_filepaths:
        try:
            image_data = TimestampedImage_pb2.TimestampedImage()
            f = open(image_filepath,'rb')
            image_data.ParseFromString(f.read())
            f.close()
            image_tags.append(image_data)
        except:
            print("Could not read image file %s." %(image_filepath))
            continue
    filepaths = [os.path.join(motion_data_folder, f) for f in os.listdir(motion_data_folder) if os.path.isfile(os.path.join(motion_data_folder, f)) and str.lower(os.path.splitext(f)[1])==".pb"]
    for filepath in filepaths:
        try:
            data = TimestampedPacketMotionData_pb2.TimestampedPacketMotionData()
            f = open(filepath,'rb')
            data.ParseFromString(f.read())
            f.close()
            motion_packets.append(data)
        except:
            print("Could not read udp file %s." %(filepath))
            continue
motion_packets = sorted(motion_packets, key=udpPacketKey)
session_times = np.array([packet.udp_packet.m_header.m_sessionTime for packet in motion_packets])
system_times = np.array([float(packet.timestamp)/1000.0 for packet in motion_packets])
print(system_times)
print(session_times)
maxudptime = system_times[-1]
image_tags = [tag for tag in image_tags if float(tag.timestamp)/1000.0<maxudptime]
image_tags = sorted(image_tags, key = imageDataKey)
image_timestamps = np.array([float(data.timestamp)/1000.0 for data in image_tags])


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
system_times = np.array([float(packet.timestamp)/1000.0 for packet in motion_packets])

print("Range of session times: [%f,%f]" %(session_times[0], session_times[-1]))
print("Range of udp system times: [%f,%f]" %(system_times[0], system_times[-1]))
print("Range of image system times: [%f,%f]" %(image_timestamps[0], image_timestamps[-1]))

poses = [extractPose(packet.udp_packet) for packet in motion_packets]
velocities = [extractVelocity(packet.udp_packet) for packet in motion_packets]
positions = np.array([pose[0] for pose in poses])
quaternions = np.array([pose[1] for pose in poses])
if args.use_given_angular_velocities:
    angular_velocities = [extractAngularVelocity(packet.udp_packet) for packet in motion_packets]
else:
    angular_velocities = quaternion.angular_velocity(quaternions, session_times)
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
velocity_interpolant = scipy.interpolate.interp1d(session_times, velocities, axis=0, kind='linear')
interpolated_positions = position_interpolant(image_session_timestamps)
interpolated_velocities = velocity_interpolant(image_session_timestamps)
interpolated_quaternions = quaternion.squad(quaternions, session_times, image_session_timestamps)
interpolated_angular_velocities = quaternion.angular_velocity(interpolated_quaternions, image_session_timestamps)
print(len(image_tags))
print(len(image_session_timestamps))
print(len(interpolated_positions))
print(len(interpolated_quaternions))
print(len(interpolated_angular_velocities))
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
num_label_poses = args.num_label_poses
#scipy.interpolate.interp1d
for idx in range(len(image_tags)):
    imagetag = image_tags[idx]
    label_tag = PoseSequenceLabel_pb2.PoseSequenceLabel()
    label_tag.car_pose.frame = FrameId_pb2.GLOBAL
    label_tag.car_velocity.frame = FrameId_pb2.GLOBAL
    label_tag.car_angular_velocity.frame = FrameId_pb2.GLOBAL
    label_tag.image_tag.CopyFrom(imagetag)
    t_interp = image_session_timestamps[idx]
    label_tag.car_pose.session_time = t_interp
    label_tag.car_velocity.session_time = t_interp
    label_tag.car_angular_velocity.session_time = t_interp
    i = bisect.bisect_left(session_times,t_interp)
    tstart = session_times[i]
    if((tstart-t_interp)<1E-3):
        print("Found a really short iterpolation distance. starting one index forward")
        i = i + 1
    if( i==0 or i > (len(session_times)- num_label_poses) ):
        continue
    print("Image session time: %f. Bisector session time: %f. Bisector index: %d" % (t_interp, session_times[i], i))
    pos1, quat1 = poses[i-1]
    pos2, quat2 = poses[i]
    vel1 = velocities[i-1]
    vel2 = velocities[i]
    angvel1 = angular_velocities[i-1]
    angvel2 = angular_velocities[i]
    interpolated_positions
    carposition_global = interpolated_positions[idx]
    #carposition_global = interpolateVectors(pos1,session_times[i-1],pos2,session_times[i], t_interp)
    carvelocity_global = interpolated_velocities[idx]
    #carvelocity_global = interpolateVectors(vel1,session_times[i-1],vel2,session_times[i], t_interp)
    carquat_global = interpolated_quaternions[idx]
    #carquat_global = quaternion.slerp(quat1,quat2,session_times[i-1],session_times[i], t_interp)
    carangvelocity_global = interpolated_angular_velocities[idx]
    #carangvelocity_global = interpolateVectors(angvel1,session_times[i-1],angvel2,session_times[i], t_interp)
        
    label_tag.car_pose.translation.x = carposition_global[0]
    label_tag.car_pose.translation.y = carposition_global[1]
    label_tag.car_pose.translation.z = carposition_global[2]
    label_tag.car_pose.rotation.x = carquat_global.x
    label_tag.car_pose.rotation.y = carquat_global.y
    label_tag.car_pose.rotation.z = carquat_global.z
    label_tag.car_pose.rotation.w = carquat_global.w

    label_tag.car_velocity.vector.x = carvelocity_global[0]
    label_tag.car_velocity.vector.y = carvelocity_global[1]
    label_tag.car_velocity.vector.z = carvelocity_global[2]
    
    label_tag.car_angular_velocity.vector.x = carangvelocity_global[0]
    label_tag.car_angular_velocity.vector.y = carangvelocity_global[1]
    label_tag.car_angular_velocity.vector.z = carangvelocity_global[2]

    carpose_global = toHomogenousTransform(carposition_global, carquat_global)
    #yes, I know this is an un-necessary inverse computation. Sue me.
    carposeinverse_global = la.inv(carpose_global)
    #print()
    #print()
    #print(carposition_global)
    #print(carpose_global)
    #print()
    for j in range(args.num_label_poses):
        # label_tag.
        pose_forward_pb = Pose3d_pb2.Pose3d()
        velocity_forward_pb = Vector3dStamped_pb2.Vector3dStamped()
        angular_velocity_forward_pb = Vector3dStamped_pb2.Vector3dStamped()
        pose_forward_pb.frame = FrameId_pb2.LOCAL
        velocity_forward_pb.frame = FrameId_pb2.LOCAL
        angular_velocity_forward_pb.frame = FrameId_pb2.LOCAL
        forward_index = i+j
        packet_forward = motion_packets[forward_index].udp_packet
        pos_forward_global, quat_forward_global = poses[forward_index]
        angular_velocity_global = angular_velocities[forward_index]
        velocity_global = velocities[forward_index]
        pose_forward_global = toHomogenousTransform(pos_forward_global, quat_forward_global)
        pose_forward_local = np.matmul(carposeinverse_global, pose_forward_global)
        velocity_local = np.matmul(carposeinverse_global[0:3,0:3], velocity_global)
        angular_velocity_local = np.matmul(carposeinverse_global[0:3,0:3], angular_velocity_global)
        position_forward_local, quat_forward_local = fromHomogenousTransform(pose_forward_local)
        #print()
        #print(pose_forward_local)
        #print(position_forward_local)
        #print(la.norm(pos_forward_global - carpos_global))
        #print(la.norm(position_forward_numpy_local))
        pose_forward_pb.translation.x = position_forward_local[0]
        pose_forward_pb.translation.y = position_forward_local[1]
        pose_forward_pb.translation.z = position_forward_local[2]
        pose_forward_pb.rotation.x = quat_forward_local.x
        pose_forward_pb.rotation.y = quat_forward_local.y
        pose_forward_pb.rotation.z = quat_forward_local.z
        pose_forward_pb.rotation.w = quat_forward_local.w

        velocity_forward_pb.vector.x = velocity_local[0]
        velocity_forward_pb.vector.y = velocity_local[1]
        velocity_forward_pb.vector.z = velocity_local[2]
        
        angular_velocity_forward_pb.vector.x = angular_velocity_local[0]
        angular_velocity_forward_pb.vector.y = angular_velocity_local[1]
        angular_velocity_forward_pb.vector.z = angular_velocity_local[2]

        pose_forward_pb.session_time = packet_forward.m_header.m_sessionTime
        velocity_forward_pb.session_time = packet_forward.m_header.m_sessionTime
        angular_velocity_forward_pb.session_time = packet_forward.m_header.m_sessionTime
        label_tag.subsequent_poses.append(pose_forward_pb)
        label_tag.subsequent_linear_velocities.append(velocity_forward_pb)
        label_tag.subsequent_angular_velocities.append(angular_velocity_forward_pb)
    #print()
    #print()
    label_tag_JSON = google.protobuf.json_format.MessageToJson(label_tag, including_default_value_fields=True)
    image_file_base = os.path.splitext(os.path.split(label_tag.image_tag.image_file)[1])[0]
    label_tag_file_path = os.path.join(image_folder,image_file_base + "_sequence_label.json")
    f = open(label_tag_file_path,'w')
    f.write(label_tag_JSON)
    f.close()
    #print(carquatinverse)
   # print(carquat)





