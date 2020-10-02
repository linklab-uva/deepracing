import numpy as np
import numpy.linalg as la
import scipy
import skimage
import PIL
from PIL import Image as PILImage
from deepracing.exceptions import DeepRacingException
import TimestampedPacketMotionData_pb2
import PoseSequenceLabel_pb2
import TimestampedImage_pb2
import MultiAgentLabel_pb2
import Vector3dStamped_pb2
import Trajectory_pb2
import argparse
import os
import google.protobuf.json_format
import Pose3d_pb2
import cv2
import bisect
import FrameId_pb2
import scipy.interpolate
import deepracing.backend
import deepracing.pose_utils as pose_utils
import deepracing.protobuf_utils as proto_utils
from deepracing.protobuf_utils import getAllSessionPackets, getAllImageFilePackets, getAllMotionPackets, extractPose, extractVelocity, extractPosition, extractRotation
from tqdm import tqdm as tqdm
import yaml
import shutil
import Spline3D_pb2
from scipy.spatial.kdtree import KDTree
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import RotationSpline as RotSpline
from scipy.spatial.transform import Slerp
from typing import List
import matplotlib.pyplot as plt

def imageDataKey(data):
    return data.timestamp
def udpPacketKey(packet):
    return packet.udp_packet.m_header.m_sessionTime

def poseSequenceLabelKey(label):
    return label.car_pose.session_time

parser = argparse.ArgumentParser()
parser.add_argument("db_path", help="Path to root directory of DB",  type=str)
parser.add_argument("lookahead_time", help="Amount of time to look forward for each driver",  type=float)
parser.add_argument("--sample_indices", help="Number of indices to sample the spline for on the region specified by lookahead_time",  type=int, default=60)
parser.add_argument("--max_distance", help="Ignore other agents further than this many meters away",  type=float, default=60)
parser.add_argument("--spline_degree", help="Spline degree to fit",  type=int, default=3)
parser.add_argument("--debug", help="Display debug plots", action="store_true", required=False)
parser.add_argument("--output_dir", help="Output directory for the labels. relative to the database images folder",  default="multi_agent_labels", required=False)
parser.add_argument("--override", help="Always delete existing directories without prompting the user", action="store_true")
parser.add_argument("--all_agents", help="Store trajectories for all agents, not just ones near the ego vehicle", action="store_true")


args = parser.parse_args()
lookahead_time = args.lookahead_time
sample_indices = args.sample_indices
splk = args.spline_degree
root_dir = args.db_path
debug = args.debug
max_distance = args.max_distance
override = args.override
all_agents = args.all_agents
motion_data_folder = os.path.join(root_dir,"udp_data","motion_packets")
image_folder = os.path.join(root_dir,"images")
session_folder = os.path.join(root_dir,"udp_data","session_packets")

with open(os.path.join(root_dir,"f1_dataset_config.yaml"),"r") as f:
    dset_config = yaml.load(f, Loader=yaml.SafeLoader)

use_json = dset_config["use_json"]
session_packets = getAllSessionPackets(session_folder, use_json)
output_dir = os.path.join(root_dir, args.output_dir)
if os.path.isdir(output_dir):
    if override:
        shutil.rmtree(output_dir)
    else:
        s = 'asdf'
        while not ( (s=='y') or (s=='n') ):
            s = input("Directory %s already exists. overwrite it? (y\\n)" %(output_dir,))
        if s=='y':
            shutil.rmtree(output_dir)
        else:
            print("Thanks for playing!")
            exit(0)
os.makedirs(output_dir)

image_tags = getAllImageFilePackets(image_folder, use_json)
motion_packets = getAllMotionPackets(motion_data_folder, use_json)
image_tags, image_session_timestamps, motion_packets, slope, intercept = pose_utils.registerImagesToMotiondata(motion_packets,image_tags)
image_system_timestamps = np.array([data.timestamp/1000.0 for data in image_tags])

motion_packet_session_times = np.array([packet.udp_packet.m_header.m_sessionTime for packet in motion_packets])
motion_packet_system_times = np.array([packet.timestamp/1000.0 for packet in motion_packets])
tmin = motion_packet_session_times[0]
tmax = motion_packet_session_times[-1]
print("Range of motion packet session times: [%f,%f]" % (tmin,tmax))
player_car_indices = [packet.udp_packet.m_header.m_playerCarIndex for packet in motion_packets]

spectating_flags = [bool(packet.udp_packet.m_isSpectating) for packet in session_packets]
spectating = any(spectating_flags)
spectator_car_indices = [int(packet.udp_packet.m_spectatorCarIndex) for packet in session_packets]
spectator_car_indices_set = set(spectator_car_indices)
if spectating:
    print("This is a spectated dataset")
    if len(spectator_car_indices_set)>1:
        raise ValueError("Spectated datasets are only supported if you only spectate 1 car the entire time.")
    else:
        ego_vehicle_index = spectator_car_indices[0]
else:
    print("This is an ego-centric dataset")
    if len(set(player_car_indices))>1:
        raise ValueError("Got more than 1 player car index in a non-spectated dataset.")
    else:
        ego_vehicle_index = player_car_indices[0]


vehicle_positions = np.stack([np.array([extractPosition(packet.udp_packet, car_index=i) for i in range(20) ]) for packet in motion_packets], axis=0).transpose(1,0,2)
print("vehicle_positions shape: %s" % (str(vehicle_positions.shape),))
vehicle_position_diffs = vehicle_positions[:,1:] - vehicle_positions[:,:-1]
vehicle_position_diff_norms = np.linalg.norm(vehicle_position_diffs,ord=2,axis=2)
vehicle_position_diff_totals = np.sum(vehicle_position_diff_norms,axis=1)
dead_cars = vehicle_position_diff_totals<(10*lookahead_time)
print("dead_cars shape: %s" % (str(dead_cars.shape),))
# print("vehicle_position_diff_norms shape: %s" % (str(vehicle_position_diff_norms.shape),))
# legit_indices = vehicle_position_diff_norms<10.0
# first_point_true = np.array([[True for asdf in range(20)]]).transpose()
# print("first_point_true shape: %s" % (str(first_point_true.shape),))
# legit_indices = np.concatenate([first_point_true, legit_indices], axis=1)
# print("legit_indices shape: %s" % (str(legit_indices.shape),))
# vehicle_positions = vehicle_positions[legit_indices]
# print("vehicle_positions shape: %s" % (str(vehicle_positions.shape),))

vehicle_velocities = np.stack([np.array([extractVelocity(packet.udp_packet, car_index=i) for i in range(20) ]) for packet in motion_packets], axis=0).transpose(1,0,2)

vehicle_quaternions = np.stack([np.array([extractRotation(packet.udp_packet, car_index=i) for i in range(20) ]) for packet in motion_packets], axis=0).transpose(1,0,2)

# vehicle_velocities = vehicle_velocities[legit_indices]
# vehicle_quaternions = vehicle_quaternions[legit_indices]

try:
    fig = plt.figure()
    allx = vehicle_positions[:,:,0]
    plt.xlim(np.max(allx)+10.0, np.min(allx)-10.0)
    plt.plot(vehicle_positions[ego_vehicle_index,:,0], vehicle_positions[ego_vehicle_index,:,2], label="Ego vehicle path")
    for i in range(20):
        if i!=ego_vehicle_index and (not dead_cars[i]):
            plt.scatter(vehicle_positions[i,:,0], vehicle_positions[i,:,2])
    plt.show()
except KeyboardInterrupt:
  exit(0)
except Exception as e:
  raise e
  print("Skipping visualiation")
  print(e)
print("Fitting position interpolants")
vehicle_position_interpolants : List[scipy.interpolate.BSpline] = [scipy.interpolate.make_interp_spline(motion_packet_session_times, vehicle_positions[i], k=splk) for i in range(20)]
ego_vehicle_position_interpolant : scipy.interpolate.BSpline = vehicle_position_interpolants[ego_vehicle_index]
# print("Fitting position KDTrees")
# vehicle_kd_trees : List[KDTree] = [KDTree(vehicle_positions[i]) for i in range(20)]
print("Fitting velocity interpolants")
vehicle_velocity_interpolants : List[scipy.interpolate.BSpline] = [scipy.interpolate.make_interp_spline(motion_packet_session_times, vehicle_velocities[i], k=splk) for i in range(20)]
ego_vehicle_velocity_interpolant : scipy.interpolate.BSpline = vehicle_velocity_interpolants[ego_vehicle_index]
print("Fitting ego vehicle rotation interpolant")
ego_vehicle_rotation_interpolant : RotSpline = RotSpline(motion_packet_session_times, Rot.from_quat(vehicle_quaternions[ego_vehicle_index]))

print("Creating LMDB")
lmdb_dir = os.path.join(output_dir,"lmdb")
if os.path.isdir(lmdb_dir):
    shutil.rmtree(lmdb_dir)
os.makedirs(lmdb_dir)
print("Generating interpolated labels")
config_dict : dict = {"lookahead_time": lookahead_time, "regression_slope": float(slope), "regression_intercept": float(intercept), "sample_indices":sample_indices}

with open(os.path.join(output_dir,'config.yaml'), 'w') as yaml_file:
    yaml.dump(config_dict, yaml_file)#, Dumper=yaml.SafeDumper)
db = deepracing.backend.MultiAgentLabelLMDBWrapper()
db.openDatabase( lmdb_dir, mapsize=int(round(76000*len(image_tags)*1.25)) , max_spare_txns=16, readonly=False, lock=True )
for idx in tqdm(range(len(image_tags))):
    label_tag = MultiAgentLabel_pb2.MultiAgentLabel()
    try:
        label_tag.image_tag.CopyFrom(image_tags[idx])
        label_tag.ego_car_index = ego_vehicle_index
        tlabel = image_session_timestamps[idx]
        if (tlabel < (tmin + 2.0*lookahead_time)) or (tlabel > (tmax - 2.0*lookahead_time)):
            continue
        
        label_tag.ego_agent_pose.frame = FrameId_pb2.GLOBAL
        label_tag.ego_agent_linear_velocity.frame = FrameId_pb2.GLOBAL
        label_tag.ego_agent_angular_velocity.frame = FrameId_pb2.GLOBAL
        label_tag.ego_agent_pose.session_time = tlabel
        label_tag.ego_agent_linear_velocity.session_time = tlabel
        label_tag.ego_agent_angular_velocity.session_time = tlabel

        ego_vehicle_position = ego_vehicle_position_interpolant(tlabel)
        label_tag.ego_agent_pose.translation.CopyFrom(proto_utils.vectorFromNumpy(ego_vehicle_position))

        ego_vehicle_rotation = ego_vehicle_rotation_interpolant(tlabel)
        label_tag.ego_agent_pose.rotation.CopyFrom(proto_utils.quaternionFromScipy(ego_vehicle_rotation))

        ego_vehicle_linear_velocity = ego_vehicle_velocity_interpolant(tlabel)
        label_tag.ego_agent_linear_velocity.vector.CopyFrom(proto_utils.vectorFromNumpy(ego_vehicle_linear_velocity))
        label_tag.ego_agent_linear_velocity.frame = FrameId_pb2.GLOBAL
        label_tag.ego_agent_linear_velocity.session_time = tlabel

        ego_vehicle_angular_velocity = ego_vehicle_rotation_interpolant(tlabel,1)
        label_tag.ego_agent_angular_velocity.vector.CopyFrom(proto_utils.vectorFromNumpy(ego_vehicle_angular_velocity))
        label_tag.ego_agent_angular_velocity.frame = FrameId_pb2.GLOBAL
        label_tag.ego_agent_angular_velocity.session_time = tlabel

        ego_pose_matrix = np.eye(4)
        ego_pose_matrix[0:3,0:3] = ego_vehicle_rotation.as_matrix()
        ego_pose_matrix[0:3,3] = ego_vehicle_position
        ego_pose_matrix_inverse = np.linalg.inv(ego_pose_matrix)

        match_found = False
        match_positions = []
        tstart = tlabel
        tend = tlabel + lookahead_time
        tsamp = np.linspace(tstart, tend, num=sample_indices)
        istart = bisect.bisect_left(motion_packet_session_times, tstart-lookahead_time)
        iend = bisect.bisect_left(motion_packet_session_times, tend+lookahead_time)
        for i in range(0,20):
            positions = vehicle_positions[i]
            position_interpolant = vehicle_position_interpolants[i]
            positions_samp_global = position_interpolant(tsamp)
            dead_car = dead_cars[i] or np.sum(np.linalg.norm(positions_samp_global[1:] - positions_samp_global[0:-1], ord=2,axis=1))<(2.5*lookahead_time)
            if i==ego_vehicle_index or dead_car:
                continue
           # kdtree = vehicle_kd_trees[i]
            velocities = vehicle_velocities[i]
            quaternions = vehicle_quaternions[i]
            velocity_interpolant = vehicle_velocity_interpolants[i]


            if np.any(np.linalg.norm(quaternions[istart:iend],ord=2,axis=1)<0.9) or np.all(positions[istart:iend]==0.0) or np.any(np.linalg.norm(positions[istart:iend],ord=2,axis=1)<2.0):
                continue
            try:
                rotation_interpolant = RotSpline(motion_packet_session_times[istart:iend], Rot.from_quat(quaternions[istart:iend]))
            except Exception as e:
                raise DeepRacingException("Could not create rotation interpolation for car %d" % i)
                #continue
            velocities_samp_global = velocity_interpolant(tsamp)
            rotations_samp_global = rotation_interpolant(tsamp)

            angvel_samp_global = rotation_interpolant(tsamp,1)
            poses_global = np.array([np.eye(4) for asdf in range(tsamp.shape[0])])
            poses_global[:,0:3,0:3] = rotations_samp_global.as_matrix()
            poses_global[:,0:3,3] = positions_samp_global
            rotations_samp_global = rotation_interpolant(tsamp)

            poses_samp = np.matmul(ego_pose_matrix_inverse,poses_global)
            positions_samp = poses_samp[:,0:3,3]
            rotations_samp = Rot.from_matrix(poses_samp[:,0:3,0:3])
            quaternions_samp = rotations_samp.as_quat()
            velocities_samp = np.matmul(ego_pose_matrix_inverse[0:3,0:3],velocities_samp_global.transpose()).transpose()
            angvel_samp = np.matmul(ego_pose_matrix_inverse[0:3,0:3],angvel_samp_global.transpose()).transpose()


            carpositionlocal = positions_samp[0]
            x = carpositionlocal[0]
            y = carpositionlocal[1]
            z = carpositionlocal[2]
            match_found = (z>-1.5) and (z<40.0) and (abs(x) < 25)
            if match_found:
                match_positions.append(carpositionlocal)

                new_trajectory_pb = label_tag.other_agent_trajectories.add()
                label_tag.trajectory_car_indices.append(i)
                for j in range(tsamp.shape[0]):
                    newpose = new_trajectory_pb.poses.add()
                    newpose.frame = FrameId_pb2.LOCAL
                    newpose.session_time = tsamp[j]
                    newpose.translation.CopyFrom(proto_utils.vectorFromNumpy(positions_samp[j]))
                    newpose.rotation.CopyFrom(proto_utils.quaternionFromScipy(rotations_samp[j]))

                    newvel = new_trajectory_pb.linear_velocities.add()
                    newvel.frame = FrameId_pb2.LOCAL
                    newvel.session_time = tsamp[j]
                    newvel.vector.CopyFrom(proto_utils.vectorFromNumpy(velocities_samp[j]))

                    newangvel = new_trajectory_pb.angular_velocities.add()
                    newangvel.frame = FrameId_pb2.LOCAL
                    newangvel.session_time = tsamp[j]
                    newvel.vector.CopyFrom(proto_utils.vectorFromNumpy(angvel_samp[j]))
        if debug and (not len(label_tag.other_agent_trajectories)==0):
            image_file = "image_%d.jpg" % idx
            print("Found a match for %s. Metadata:" %(image_file,))
            print("Match positions: " + str(match_positions))
            print("Closest packet index: " + str(istart))
            print("Car indices: " + str(label_tag.trajectory_car_indices))
            # imagein = cv2.imread(os.path.join(image_folder,image_file))
            # cv2.imshow("Image",imagein)
            # cv2.waitKey(0)
            



        
        
            
            
    except KeyboardInterrupt as e:
        break
    except DeepRacingException as e: 
        print("Could not generate label for %s" %(label_tag.image_tag.image_file))
        print("Exception message: %s"%(str(e)))
        continue       
    image_file_base = os.path.splitext(os.path.basename(label_tag.image_tag.image_file))[0]
    key = image_file_base

    label_tag_json_file = os.path.join(output_dir, key + "_multi_agent_label.json")
    with open(label_tag_json_file,'w') as f: 
        label_tag_JSON = google.protobuf.json_format.MessageToJson(label_tag, including_default_value_fields=True, indent=2)
        f.write(label_tag_JSON)

    label_tag_binary_file = os.path.join(output_dir, key + "_multi_agent_label.pb")
    with open(label_tag_binary_file,'wb') as f: 
        f.write( label_tag.SerializeToString() )
    db.writeMultiAgentLabel( key , label_tag )
    if debug and len(label_tag.other_agent_trajectories)!=0:# and idx%30==0:
        fig1 = plt.subplot(1, 2, 1)
        imcv = cv2.imread(os.path.join(image_folder, label_tag.image_tag.image_file), cv2.IMREAD_UNCHANGED)
        plt.imshow(cv2.cvtColor(imcv, cv2.COLOR_BGR2RGB))

        fig2 = plt.subplot(1, 2, 2)
        # ego_label_positions = np.array([[pose.translation.x,pose.translation.y,pose.translation.z]  for pose in label_tag.ego_agent_trajectory.poses ])
        # print("Ego position 0: " + str(ego_label_positions[0]))
        # plt.plot(-ego_label_positions[:,0], ego_label_positions[:,2], label="Ego Vehicle Trajectory", c="b")
        # plt.legend()
        label_positions = np.array([ np.array([[pose.translation.x, pose.translation.y, pose.translation.z]  for pose in agent_trajectory.poses ]).transpose() for agent_trajectory in label_tag.other_agent_trajectories])
        minx = np.min(label_positions[:,0])-5.0
        maxx = np.max(label_positions[:,0])+5.0
        maxz = np.max(label_positions[:,2])+5.0
        plt.xlim(maxx,minx)
        plt.ylim(0,maxz)
        for k in range(label_positions.shape[0]):
            agent_trajectory = label_positions[k]
            #label_positions_local = np.matmul(ego_pose_matrix_inverse, label_positions)
            plt.plot(agent_trajectory[0], agent_trajectory[2])#, label="Ego Vehicle Trajectory", c="b")
        plt.show()


print("Loading database labels.")
db_keys = db.getKeys()
label_pb_tags = []
for i,key in tqdm(enumerate(db_keys), total=len(db_keys)):
    #print(key)
    label_pb_tags.append(db.getMultiAgentLabel(key))
    if(not (label_pb_tags[-1].image_tag.image_file == key+".jpg")):
        raise AttributeError("Mismatch between database key: %s and associated image file: %s" %(key, label_pb_tags.image_tag.image_file))
sorted_indices = np.argsort( np.array([lbl.ego_agent_pose.session_time for lbl in label_pb_tags]) )
#tagscopy = label_pb_tags.copy()
label_pb_tags_sorted = [label_pb_tags[sorted_indices[i]] for i in range(len(sorted_indices))]
sorted_keys = []
print("Checking for invalid keys.")
for packet in tqdm(label_pb_tags_sorted):
    #print(key)
    key = os.path.splitext(packet.image_tag.image_file)[0]
    try:
        lbl = db.getMultiAgentLabel(key)
        sorted_keys.append(key)
    except:
        print("Skipping bad key: %s" %(key))
        continue
key_file = os.path.join(root_dir,"multi_agent_label_keys.txt")
with open(key_file, 'w') as filehandle:
    filehandle.writelines("%s\n" % key for key in sorted_keys)    





