from typing import List
import numpy as np
import numpy.linalg as la
from PIL import Image as PILImage
import TimestampedPacketMotionData_pb2, TimestampedPacketCarTelemetryData_pb2
import argparse
import os
import scipy.interpolate
import sklearn.decomposition
import deepracing.backend
import deepracing.pose_utils
import deepracing.protobuf_utils
from tqdm import tqdm as tqdm
import yaml
import matplotlib.pyplot as plt


def udpPacketKey(packet):
    return packet.udp_packet.m_header.m_sessionTime

parser = argparse.ArgumentParser()
parser.add_argument("db_path", help="Path to calibration data",  type=str)
args = parser.parse_args()
argdict = vars(args)
rootpath = argdict["db_path"]
with open(os.path.join(rootpath, "f1_dataset_config.yaml"), "r") as f:
    dset_config = yaml.load(f, Loader=yaml.SafeLoader)
use_json = dset_config["use_json"]
udp_folder = os.path.join(rootpath,dset_config["udp_folder"])

motion_folder = os.path.join(udp_folder, "motion_packets")
session_folder = os.path.join(udp_folder, "session_packets")
telemetry_folder = os.path.join(udp_folder, "car_telemetry_packets")

session_packets = deepracing.protobuf_utils.getAllSessionPackets(session_folder, use_json)
spectating_flags = [bool(packet.udp_packet.m_isSpectating) for packet in session_packets]
spectating = np.any(spectating_flags)
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

telemetry_packets : List[TimestampedPacketCarTelemetryData_pb2.TimestampedPacketCarTelemetryData] = sorted(deepracing.protobuf_utils.getAllTelemetryPackets(telemetry_folder, use_json), key=udpPacketKey)
telemetry_session_times = np.array([packet.udp_packet.m_header.m_sessionTime for packet in telemetry_packets])


motion_packets : List[TimestampedPacketMotionData_pb2.TimestampedPacketMotionData] = sorted(deepracing.protobuf_utils.getAllMotionPackets(motion_folder, use_json), key=udpPacketKey)
motion_session_times = np.array([packet.udp_packet.m_header.m_sessionTime for packet in motion_packets])
position_vectors = np.asarray([deepracing.protobuf_utils.extractPosition(packet.udp_packet) for packet in motion_packets])
position_pca = sklearn.decomposition.PCA(n_components=2)
position_pca.fit(position_vectors)
position_vectors=position_pca.inverse_transform(position_pca.transform(position_vectors))
position_spline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(motion_session_times, position_vectors, k=3)
# velocityspline : scipy.interpolate.BSpline = position_spline.derivative()
# velocity_vectors = velocityspline(motion_session_times)

velocity_vectors = np.asarray([deepracing.protobuf_utils.extractVelocity(packet.udp_packet) for packet in motion_packets])
velocity_pca = sklearn.decomposition.PCA(n_components=2)
velocity_pca.fit(velocity_vectors)
velocity_vectors=velocity_pca.inverse_transform(velocity_pca.transform(velocity_vectors))
velocityspline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(motion_session_times, velocity_vectors, k=1)

speeds = np.linalg.norm(velocity_vectors, ord=2, axis=1)
idx = speeds>18.5
motion_session_times = motion_session_times[idx]
velocity_vectors = velocity_vectors[idx]
speeds = speeds[idx]
speedspline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(motion_session_times, speeds, k=3)
accelerationspline : scipy.interpolate.BSpline = speedspline.derivative()
linear_accelerations = accelerationspline(motion_session_times)

unit_tangents = velocity_vectors/speeds[:,np.newaxis]
# accelerationspline : scipy.interpolate.BSpline = velocityspline.derivative()
# acceleration_vectors = accelerationspline(motion_session_times)
# centripetal_acceleration_vectors = acceleration_vectors - linear_accelerations[:,np.newaxis]*unit_tangents
# centripetal_accelerations = np.linalg.norm(centripetal_acceleration_vectors, ord=2, axis=1)
# linear_accelerations = (np.asarray([packet.udp_packet.m_carMotionData[packet.udp_packet.m_header.m_playerCarIndex].m_gForceLateral for packet in motion_packets])*9.81)[idx]

speedfig = plt.figure()
plt.plot(motion_session_times, speeds)
plt.xlabel("Session Time (s)")
plt.ylabel("Speed (m/s)")

idxaccel = linear_accelerations>=0.0
accelfig = plt.figure()
plt.scatter(speeds[idxaccel], linear_accelerations[idxaccel])
plt.xlabel("Speeds (m/s)")
plt.ylabel("Linear Acceleration (m/s^2)")

idxbrake = np.logical_not(idxaccel)
brakefig = plt.figure()
plt.scatter(speeds[idxbrake], linear_accelerations[idxbrake])
plt.xlabel("Speeds (m/s)")
plt.ylabel("Linear Braking (m/s^2)")

plt.show()