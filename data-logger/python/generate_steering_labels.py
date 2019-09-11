import numpy as np
import numpy.linalg as la
import quaternion
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
import scipy
import scipy.interpolate
import deepracing.pose_utils
import deepracing.backend
from deepracing.pose_utils import getAllImageFilePackets, getAllMotionPackets
from deepracing.protobuf_utils import getAllSessionPackets, getAllTelemetryPackets
from tqdm import tqdm as tqdm
import yaml
import LabeledImage_pb2
def imageDataKey(data):
    return data.timestamp
def udpPacketKey(packet):
    return packet.timestamp

parser = argparse.ArgumentParser()
parser.add_argument("db_path", help="Path to root directory of DB",  type=str)
parser.add_argument("--assume_linear_timescale", help="Assumes the slope between system time and session time is 1.0", action="store_true", required=False)
parser.add_argument("--json", help="Assume dataset files are in JSON rather than binary .pb files.",  action="store_true", required=False)
parser.add_argument("--output_dir", help="Output directory for the labels. relative to the database images folder",  default="steering_labels", required=False)
args = parser.parse_args()
telemetry_folder = os.path.join(args.db_path,"udp_data","car_telemetry_packets")
image_folder = os.path.join(args.db_path,"images")
session_folder = os.path.join(args.db_path,"udp_data","session_packets")
session_packets = getAllSessionPackets(session_folder,args.json)

spectating_flags = [bool(packet.udp_packet.m_isSpectating) for packet in session_packets]
spectating = False
for flag in spectating_flags:
    spectating = spectating or flag
car_indices = [int(packet.udp_packet.m_spectatorCarIndex) for packet in session_packets]
print(spectating_flags)
print(car_indices)
print(spectating)
car_indices_set = set(car_indices)
car_index = 0
if spectating:
    if len(car_indices_set)>1:
        raise ValueError("Spectated datasets are only supported if you only spectate 1 car the entire time.")
    else:
        car_index = car_indices[0]

image_tags = deepracing.pose_utils.getAllImageFilePackets(image_folder, args.json)
telemetry_packets = deepracing.protobuf_utils.getAllTelemetryPackets(telemetry_folder, args.json)
telemetry_packets = sorted(telemetry_packets, key=udpPacketKey)
session_times = np.array([packet.udp_packet.m_header.m_sessionTime for packet in telemetry_packets])
system_times = np.array([packet.timestamp/1000.0 for packet in telemetry_packets])
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

telemetry_packets = telemetry_packets[firstIndex:]
telemetry_packets = sorted(telemetry_packets, key=udpPacketKey)
session_times = np.array([packet.udp_packet.m_header.m_sessionTime for packet in telemetry_packets])
unique_session_times, unique_session_time_indices = np.unique(session_times, return_index=True)
telemetry_packets = [telemetry_packets[i] for i in unique_session_time_indices]
telemetry_packets = sorted(telemetry_packets, key=udpPacketKey)
session_times = np.array([packet.udp_packet.m_header.m_sessionTime for packet in telemetry_packets])
system_times = np.array([packet.timestamp/1000.0 for packet in telemetry_packets])

print("Range of session times: [%f,%f]" %(session_times[0], session_times[-1]))
print("Range of udp system times: [%f,%f]" %(system_times[0], system_times[-1]))
print("Range of image system times: [%f,%f]" %(image_timestamps[0], image_timestamps[-1]))

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
telemetry_data = [packet.udp_packet for packet in telemetry_packets]
steering = [float(data.m_carTelemetryData[car_index].m_steer)/100.0 for data in telemetry_data]
throttle = [float(data.m_carTelemetryData[car_index].m_throttle)/100.0 for data in telemetry_data]
brake = [float(data.m_carTelemetryData[car_index].m_brake)/100.0 for data in telemetry_data]
steering_interpolant = scipy.interpolate.interp1d(session_times, steering , axis=0, kind='cubic')
throttle_interpolant = scipy.interpolate.interp1d(session_times, throttle , axis=0, kind='cubic')
brake_interpolant = scipy.interpolate.interp1d(session_times, brake , axis=0, kind='cubic')

interpolated_steerings = steering_interpolant(image_session_timestamps)
interpolated_throttles = throttle_interpolant(image_session_timestamps)
interpolated_brakes = brake_interpolant(image_session_timestamps)

print()
print(len(image_tags))
print(len(image_session_timestamps))
print()
print("Linear map from system time to session time: session_time = %f*system_time + %f" %(slope,intercept))
print("Standard error: %f" %(std_err))
print("R^2: %f" %(r_value**2))
try:
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
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
except KeyboardInterrupt:
    exit(0)
except:
  text = input("Could not import matplotlib, skipping visualization. Enter anything to continue.")
#scipy.interpolate.interp1d
output_dir=os.path.join(image_folder, args.output_dir)
lmdb_dir=os.path.join(image_folder, args.output_dir+"_lmdb")
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
if not os.path.isdir(lmdb_dir):
    os.makedirs(lmdb_dir)
lmdb_backend = deepracing.backend.ControlLabelLMDBWrapper()
lmdb_backend.readDatabase(lmdb_dir, mapsize=3e9, readonly=False)
print("Generating interpolated labels")
for idx in tqdm(range(len(image_tags))):
    image_tag = image_tags[idx]
    label_tag = LabeledImage_pb2.LabeledImage()
    label_tag.image_file = image_tag.image_file
    label_tag.label.steering = interpolated_steerings[idx]
    label_tag.label.throttle = interpolated_throttles[idx]
    label_tag.label.brake = interpolated_brakes[idx]
    label_tag_JSON = google.protobuf.json_format.MessageToJson(label_tag, including_default_value_fields=True)
    image_file_base = os.path.splitext(os.path.split(label_tag.image_file)[1])[0]
    label_tag_file_path = os.path.join(output_dir, image_file_base + "_control_label.json")
    f = open(label_tag_file_path,'w')
    f.write(label_tag_JSON)
    f.close()
    label_tag_file_path_binary = os.path.join(output_dir, image_file_base + "_control_label.pb")
    f = open(label_tag_file_path_binary,'wb')
    f.write(label_tag.SerializeToString())
    f.close()
    key = image_file_base
    lmdb_backend.writeControlLabel(key,label_tag)
lmdb_backend.readDatabase(lmdb_dir, mapsize=3e9, readonly=True)
keys = [os.path.splitext(os.path.split(image_tag.image_file)[1])[0] for image_tag in image_tags]
irand = np.random.randint(0,high=len(keys))
keyrand = keys[irand]
print("Entry at key %s" % (keyrand))
print(lmdb_backend.getControlLabel(keyrand))
key_file = os.path.join(args.db_path,"controloutputkeys.txt")
with open(key_file, 'w') as filehandle:
    filehandle.writelines("%s\n" % key for key in keys)
    





