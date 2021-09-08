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
from deepracing.protobuf_utils import getAllSessionPackets, getAllImageFilePackets, getAllMotionPackets, extractPose, extractVelocity, extractAngularVelocity
from tqdm import tqdm as tqdm
import yaml
import shutil
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import RotationSpline as RotSpline
import json 
from scipy.spatial import KDTree as KDTree
import matplotlib.pyplot as plt
import time
from deepracing import trackNames
import json
def imageDataKey(data):
    return data.timestamp

def poseSequenceLabelKey(label):
    return label.car_pose.session_time

parser = argparse.ArgumentParser()
parser.add_argument("db_path", help="Path to root directory of DB",  type=str)
parser.add_argument("--output_dir", help="Output directory for the labels. relative to the database folder",  default="image_poses", required=False)



args = parser.parse_args()


root_dir = args.db_path
with open(os.path.join(root_dir,"f1_dataset_config.yaml"),"r") as f:
    config = yaml.load(f,Loader=yaml.SafeLoader)
    use_json = config["use_json"]
motion_data_folder = os.path.join(root_dir,"udp_data","motion_packets")
image_folder = os.path.join(root_dir,"images")
session_folder = os.path.join(root_dir,"udp_data","session_packets")
session_packets = getAllSessionPackets(session_folder,use_json)
track_ids = [packet.udp_packet.m_trackId for packet in session_packets]
if(len(list(set(track_ids))) > 1):
    raise ValueError("This script only works on sessions where the whole session was done on the same track.")
track_id = track_ids[0]
output_dir = os.path.join(root_dir, args.output_dir)
if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
time.sleep(1.0)
os.makedirs(output_dir)


spectating_flags = [bool(packet.udp_packet.m_isSpectating) for packet in session_packets]
spectating = any(spectating_flags)
car_indices = [int(packet.udp_packet.m_spectatorCarIndex) for packet in session_packets]

car_indices_set = set(car_indices)
print(car_indices_set)
print(car_indices)
if spectating:
    if len(car_indices_set)>1:
        raise ValueError("Spectated datasets are only supported if you only spectate 1 car the entire time.")
    else:
        car_index = car_indices[0]
else:
    car_index = None

image_tags = getAllImageFilePackets(image_folder, use_json)
motion_packets = getAllMotionPackets(motion_data_folder, use_json)
motion_packets = sorted(motion_packets, key=deepracing.timestampedUdpPacketKey)
session_times = np.array([packet.udp_packet.m_header.m_sessionTime for packet in motion_packets])
system_times = np.array([packet.timestamp/1000.0 for packet in motion_packets])
print(system_times)
print(session_times)
maxudptime = system_times[-1]
image_tags = [ tag for tag in image_tags if tag.timestamp/1000.0<(maxudptime) ]
image_tags = sorted(image_tags, key = imageDataKey)
image_timestamps = np.array([data.timestamp/1000.0 for data in image_tags])


first_image_time = image_timestamps[0]
print(first_image_time)
Imin = system_times>(first_image_time + 1.0)
firstIndex = np.argmax(Imin)

motion_packets = motion_packets[firstIndex:]
motion_packets = sorted(motion_packets, key=deepracing.timestampedUdpPacketKey)
session_times = np.array([packet.udp_packet.m_header.m_sessionTime for packet in motion_packets], dtype=np.float64)
unique_session_times, unique_session_time_indices = np.unique(session_times, return_index=True)
motion_packets = [motion_packets[i] for i in unique_session_time_indices]
motion_packets = sorted(motion_packets, key=deepracing.timestampedUdpPacketKey)
session_times = np.array([packet.udp_packet.m_header.m_sessionTime for packet in motion_packets], dtype=np.float64)
system_times = np.array([packet.timestamp/1000.0 for packet in motion_packets], dtype=np.float64)

print("Range of session times: [%f,%f]" %(session_times[0], session_times[-1]))
print("Range of udp system times: [%f,%f]" %(system_times[0], system_times[-1]))
print("Range of image system times: [%f,%f]" %(image_timestamps[0], image_timestamps[-1]))

poses = [extractPose(packet.udp_packet, car_index=car_index) for packet in motion_packets]
velocities = np.array([extractVelocity(packet.udp_packet, car_index=car_index) for packet in motion_packets])
positions = np.array([pose[0] for pose in poses])
position_diffs = np.diff(positions, axis=0)
position_diff_norms = la.norm(position_diffs, axis=1)
print("Diff norm vector has length %d: " % (len(position_diff_norms)))

quaternions = np.array([pose[1] for pose in poses])
rotations = Rot.from_quat(quaternions)

slope_session_time_fit, intercept_session_time_fit, rvalue, pvalue, stderr = scipy.stats.linregress(np.linspace(1,session_times.shape[0],session_times.shape[0]), session_times)
print("Slope and intercept of raw session times: [%f,%f]" %(slope_session_time_fit, intercept_session_time_fit))

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(system_times, session_times)
print("Slope and intercept of session time vs system time: [%f,%f]" %(slope, intercept))
print( "r value of session time vs system time: %f" % ( rvalue ) )
print( "r^2 value of session time vs system time: %f" % ( rvalue**2 ) )

image_session_timestamps = slope*image_timestamps + intercept
print("Range of image session times before clipping: [%f,%f]" %(image_session_timestamps[0], image_session_timestamps[-1]))

Iclip = (image_session_timestamps>(np.min(session_times) + 1.5)) * (image_session_timestamps<(np.max(session_times) - 1.5 ))
image_tags = [image_tags[i] for i in range(len(image_session_timestamps)) if Iclip[i]]
image_session_timestamps = image_session_timestamps[Iclip]
print("Range of image session times after clipping: [%f,%f]" %(image_session_timestamps[0], image_session_timestamps[-1]))


position_interpolant = scipy.interpolate.make_interp_spline(session_times, positions)
rotation_interpolant = RotSpline(session_times, rotations)
velocity_interpolant = scipy.interpolate.make_interp_spline(session_times, velocities)


interpolated_positions = position_interpolant(image_session_timestamps)
interpolated_velocities = velocity_interpolant(image_session_timestamps)
interpolated_rotations = rotation_interpolant(image_session_timestamps)
interpolated_quaternions = interpolated_rotations.as_quat()
if spectating:
    interpolated_angular_velocities = rotation_interpolant(image_session_timestamps,order=1)
else:
    angular_velocities = np.array([extractAngularVelocity(packet.udp_packet) for packet in motion_packets])
    angular_velocity_interpolant = scipy.interpolate.make_interp_spline(session_times, angular_velocities)
    interpolated_angular_velocities = angular_velocity_interpolant(image_session_timestamps)

fig = plt.figure()
plt.scatter(interpolated_positions[:,0], interpolated_positions[:,2])
plt.show()
plt.close("all")


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


output_dict : dict = dict()
image_keys = []
for i in tqdm(range(len(image_tags))):
    image_tag = image_tags[i]
    key, extension = os.path.splitext(image_tag.image_file)
    key = key.replace("\n","")
    image_keys.append(key)
    imagedict : dict = dict()
    imagedict["position"] = interpolated_positions[i].tolist()
    imagedict["session_time"] = image_session_timestamps[i]
    imagedict["quaternion"] = interpolated_quaternions[i].tolist()
    imagedict["linear_velocity"] = interpolated_velocities[i].tolist()
    imagedict["angular_velocity"] = interpolated_angular_velocities[i].tolist()
    output_dict[key] = imagedict

output_dir = os.path.join(root_dir, args.output_dir)
if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

image_files = os.path.join(output_dir, "image_files.txt")
with open(image_files, "w") as f:
    f.writelines([key+"\n" for key in image_keys])

dictionary_file = os.path.join(output_dir, "image_poses.json")
with open(dictionary_file, "w") as f:
    json.dump(output_dict, f, indent=3)

geometric_data_file = os.path.join(output_dir, "geometric_data.npz")
with open(geometric_data_file, "wb") as f:
    np.savez(f, interpolated_positions=interpolated_positions, \
                interpolated_quaternions = interpolated_quaternions, \
                interpolated_velocities=interpolated_velocities, \
                interpolated_angular_velocities=interpolated_angular_velocities, \
                image_session_timestamps=image_session_timestamps, \
                udp_positions=positions, \
                udp_rotations=rotations.as_quat(), \
                udp_velocities=velocities, \
                udp_session_times=session_times, \
            )

