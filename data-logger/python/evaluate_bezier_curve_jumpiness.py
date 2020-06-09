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
import yaml
import shutil
import json
import time

def bcurveKey(bcurve: BezierCurve_pb2.BezierCurve):
    return bcurve.m_sessionTime
def motionPacketKey(motion_packet: PacketMotionData_pb2.PacketMotionData):
    return motion_packet.m_header.m_sessionTime
def getStatistics(data : np.ndarray):
    rtn : dict = {}
    rtn["mean"] = float(np.mean(data))
    rtn["variance"] = float(np.var(data))
    return rtn
parser = argparse.ArgumentParser()
parser.add_argument("db_path", help="Path to root directory of DB",  type=str)
# parser.add_argument("--assume_linear_timescale", help="Assumes the slope between system time and session time is 1.0", action="store_true", required=False)
# parser.add_argument("--output_dir", help="Output directory for the labels. relative to the database images folder",  default="steering_labels", required=False)
args = parser.parse_args()

db_path = args.db_path
udp_path = os.path.join(db_path,"udp_data")
results_dir = os.path.join(db_path,"results")
bezier_curve_path = os.path.join(udp_path, "bezier_curves")
motion_packet_path = os.path.join(udp_path, "motion_packets")
dset_config_file = os.path.join(db_path,"f1_dataset_config.yaml")
with open(dset_config_file,"r") as f:
    dset_config = yaml.load(f,Loader=yaml.SafeLoader)

use_json = dset_config["use_json"]
if os.path.isdir(results_dir):
    shutil.rmtree(results_dir)
time.sleep(0.25)
os.makedirs(results_dir)

motion_packets = sorted([packet.udp_packet for packet in proto_utils.getAllMotionPackets(motion_packet_path, use_json)],key=motionPacketKey)
tmin = motion_packets[0].m_header.m_sessionTime + 5.0
tmax = motion_packets[-1].m_header.m_sessionTime - 5.0

bcurves = [bc for bc in proto_utils.getAllBezierCurves(bezier_curve_path, use_json) if (bc.m_sessionTime < tmax and bc.m_sessionTime > tmin)]
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

average_curve_distances = []
isjump = []
distances_from_previous_pose = []
position_deltas = []
control_point_zero_deltas = []
bc_session_times = []
distance_ratios = []
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
    average_curve_distances.append(torch.mean(norms).item())
    position_prev, rotation_prev = position_spline(bcurves[i-1].m_sessionTime), rotation_spline(bcurves[i-1].m_sessionTime)
    position_curr, rotation_curr = position_spline(bcurves[i].m_sessionTime), rotation_spline(bcurves[i].m_sessionTime)

    position_deltas.append(la.norm(position_curr - position_prev))
    ball_radius = 1.0*position_deltas[-1]
   # print()

    pose_prev = np.eye(4)
    pose_prev[0:3,0:3] = rotation_prev.as_matrix()
    pose_prev[0:3,3] = position_prev

    pose_curr = np.eye(4)
    pose_curr[0:3,0:3] = rotation_curr.as_matrix()
    pose_curr[0:3,3] = position_curr

    control_point_prev = np.array([b[0,0,0], 0.0, b[0,0,1], 1.0])
    control_point_curr = np.array([b[1,0,0], 0.0, b[1,0,1], 1.0])

    control_point_prev_global = np.matmul(pose_prev, control_point_prev)[0:3]
    control_point_curr_global = np.matmul(pose_curr, control_point_curr)[0:3]
    control_point_zero_deltas.append(la.norm(control_point_prev_global - control_point_curr_global))

    distance_from_previous_pose = la.norm(control_point_curr_global - position_prev) 
    #print(distance_from_previous_pose)
    distances_from_previous_pose.append(distance_from_previous_pose)
    bc_session_times.append(bcurves[i-1].m_sessionTime)
    distance_ratios.append(distance_from_previous_pose/position_deltas[-1])
    isjump.append(distance_from_previous_pose > ball_radius)
N_points = int(len(average_curve_distances))

jumps = sum(isjump)
non_jumps = sum([not j for j in isjump])
jumpsnp = np.array(isjump,dtype=np.uint8)
data_to_write = np.column_stack((bc_session_times,position_deltas,distances_from_previous_pose,jumpsnp,distance_ratios,control_point_zero_deltas,average_curve_distances))
data_header =                    "bc_session_times,position_deltas,distances_from_previous_pose,jumpsnp,distance_ratios,control_point_zero_deltas,average_curve_distances"
processed_data_csv = os.path.join(results_dir,"processed_data.csv")
np.savetxt(processed_data_csv,data_to_write,header=data_header,delimiter=",",comments="")

processed_data_file = os.path.join(results_dir,"processed_data.json")
processed_data_dict : dict = {}
processed_data_dict["bc_session_times"]=bc_session_times
processed_data_dict["position_deltas"]=position_deltas
processed_data_dict["distances_from_previous_pose"]=distances_from_previous_pose
processed_data_dict["jumpsnp"]=jumpsnp.tolist()
processed_data_dict["distance_ratios"]=distance_ratios
processed_data_dict["control_point_zero_deltas"]=control_point_zero_deltas
processed_data_dict["average_curve_distances"]=average_curve_distances
with open(processed_data_file,"w") as f:
    json.dump(processed_data_dict,f, indent=1)


results_data_file = os.path.join(results_dir,"results.json")
results_dict : dict = {}
results_dict["position_deltas"] = getStatistics(position_deltas)
results_dict["distances_from_previous_pose"] = getStatistics(distances_from_previous_pose)
results_dict["distance_ratios"] = getStatistics(distance_ratios)
results_dict["control_point_zero_deltas"] = getStatistics(control_point_zero_deltas)
results_dict["average_curve_distances"] = getStatistics(average_curve_distances)

results_dict["jumps"] = int(jumps)
results_dict["nonjumps"] = int(non_jumps)
results_dict["numsamples"] = N_points
print(results_dict)

with open(results_data_file,"w") as f:
    json.dump(results_dict,f, indent=1)

print("Number of jumps: %d" %(jumps))
print("Number of nonjumps: %d" %(non_jumps))
print("Control Point Zero Distance mean: %f" %(np.mean(distances_from_previous_pose)))
print("Control Point Zero Distance stdev: %f" %(np.std(distances_from_previous_pose)))
figcontrolpointdistances = plt.figure()
plt.title("Histogram of distance from control point 0\n to previous position")
plt.hist(distances_from_previous_pose, bins=30)


plt.show()
    
#A = mu.bezierM()