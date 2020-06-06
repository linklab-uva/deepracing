import google.protobuf.json_format as pbjson
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import PoseSequenceLabel_pb2
import argparse
import deepracing.imutils as imutils
import torchvision
import torchvision.utils as tvutils
import torchvision.transforms.functional as tf
import torch
import BezierCurve_pb2
import TimestampedPacketMotionData_pb2
import PacketMotionData_pb2
import CarMotionData_pb2 
import deepracing.protobuf_utils as proto_utils
import deepracing_models.math_utils as mu
from scipy.spatial.transform import Rotation as Rot, RotationSpline as RotSpline
import scipy.interpolate
import numpy.linalg as la

def bcurveKey(bcurve: BezierCurve_pb2.BezierCurve):
    return bcurve.m_sessionTime
def motionPacketKey(motion_packet: PacketMotionData_pb2.PacketMotionData):
    return motion_packet.m_header.m_sessionTime
parser = argparse.ArgumentParser()
parser.add_argument("db_path", help="Path to root directory of DB",  type=str)
parser.add_argument("db_name", help="Name to assign this particular run",  type=str)
# parser.add_argument("--assume_linear_timescale", help="Assumes the slope between system time and session time is 1.0", action="store_true", required=False)
parser.add_argument("--json", help="Assume dataset files are in JSON rather than binary .pb files.",  action="store_true", required=False)
# parser.add_argument("--output_dir", help="Output directory for the labels. relative to the database images folder",  default="steering_labels", required=False)
args = parser.parse_args()

db_path = args.db_path
db_name = args.db_name
files_in_json = args.json
udp_path = os.path.join(db_path,"udp_data")
bezier_curve_path = os.path.join(udp_path, "bezier_curves")
motion_packet_path = os.path.join(udp_path, "motion_packets")

motion_packets = sorted([packet.udp_packet for packet in proto_utils.getAllMotionPackets(motion_packet_path, files_in_json)],key=motionPacketKey)
tmin = motion_packets[0].m_header.m_sessionTime + 5.0
tmax = motion_packets[-1].m_header.m_sessionTime - 5.0

bcurves = [bc for bc in proto_utils.getAllBezierCurves(bezier_curve_path, files_in_json) if (bc.m_sessionTime < tmax and bc.m_sessionTime > tmin)]
poses = [proto_utils.extractPose(packet) for packet in motion_packets]
positions = np.array([pose[0] for pose in poses])
quaternions = np.array([pose[1] for pose in poses])
wnegative = quaternions[:,3]<0
quaternions[wnegative]*=-1.0
rotations = Rot.from_quat(quaternions)
session_times = np.array([p.m_header.m_sessionTime for p in motion_packets])
position_spline = scipy.interpolate.make_interp_spline(session_times, positions)
rotation_spline = RotSpline(session_times,rotations)

bcurves = sorted(bcurves,key=bcurveKey)
bezier_order = len(bcurves[0].control_points_x)-1
t = torch.from_numpy(np.linspace(0,1,120)).unsqueeze(0).repeat(2,1).double()
M = mu.bezierM(t,bezier_order)
M = M.double()

average_norms = []
jumps = 0
non_jumps = 0
distances = []
position_deltas = []
control_point_zero_deltas = []
for i in range(1,len(bcurves)):
    b = torch.zeros(2,bezier_order+1,2)
    b[0,:,0] = torch.from_numpy(np.array(bcurves[i-1].control_points_x))
    b[0,:,1] = torch.from_numpy(np.array(bcurves[i-1].control_points_z))
    b[1,:,0] = torch.from_numpy(np.array(bcurves[i].control_points_x))
    b[1,:,1] = torch.from_numpy(np.array(bcurves[i].control_points_z))
    b = b.double()
    points_both = torch.matmul(M,b)
    deltap = (b[1] - b[0]).unsqueeze(0)
    points_delta = torch.matmul(M[0].unsqueeze(0),deltap)
    norms = torch.norm(points_delta[0],dim=1)
    average_norms.append(torch.mean(norms).item())
    position_prev, rotation_prev = position_spline(bcurves[i-1].m_sessionTime), rotation_spline(bcurves[i-1].m_sessionTime)
    position_curr, rotation_curr = position_spline(bcurves[i].m_sessionTime), rotation_spline(bcurves[i].m_sessionTime)

    position_deltas.append(la.norm(position_curr - position_prev))
    ball_radius = 1.0125*position_deltas[-1]
   # print()

    pose_prev = np.eye(4)
    pose_prev[0:3,0:3] = rotation_prev.as_matrix()
    pose_prev[0:3,3] = position_prev

    pose_curr = np.eye(4)
    pose_curr[0:3,0:3] = rotation_curr.as_matrix()
    pose_curr[0:3,3] = position_curr

    control_point_prev = np.array([b[0,0,0], 0.0, b[0,0,1], 1.0])
    control_point_curr = np.array([b[1,0,0], 0.0, b[1,0,1], 1.0])

    control_point_global = np.matmul(pose_curr, control_point_curr)
    control_point_prev_global = np.matmul(pose_prev, control_point_prev)
    control_point_zero_deltas.append(la.norm(control_point_prev_global[0:3] - control_point_global[0:3]))

    distance_from_previous_pose = la.norm(control_point_global[0:3] - position_prev) 
    distances.append(distance_from_previous_pose)
    
    if distance_from_previous_pose > ball_radius:
        jumps+=1
    else:
        non_jumps+=1
    
    

N_points = len(average_norms)
n_bins = 30

# print("Subsequent Distance mean: %f" %(np.mean(average_norms)))
# print("Subsequent Distance stdev: %f" %(np.std(average_norms)))
# figsubsequentdistances = plt.figure()
# plt.title("Histogram of mean distance between\n subsequent curves (%s)" % (db_name,))
# plt.hist(average_norms, bins=n_bins)


print("Number of jumps: %d" %(jumps))
print("Number of nonjumps: %d" %(non_jumps))
print("Control Point Zero Distance mean: %f" %(np.mean(distances)))
print("Control Point Zero Distance stdev: %f" %(np.std(distances)))
figcontrolpointdistances = plt.figure()
plt.title("Histogram of distance from control point 0\n to previous position (%s)" % (db_name,))
plt.hist(distances, bins=n_bins)


plt.show()
    
#A = mu.bezierM()