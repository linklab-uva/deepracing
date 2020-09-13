import numpy as np
import numpy.linalg as la
import scipy
import skimage
import PIL
from PIL import Image as PILImage
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
from deepracing.protobuf_utils import getAllSessionPackets, getAllImageFilePackets, getAllMotionPackets, extractPose, extractVelocity
from tqdm import tqdm as tqdm
import yaml
import shutil
import Spline3D_pb2
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import RotationSpline as RotSpline
from scipy.spatial.transform import Slerp
def imageDataKey(data):
    return data.timestamp
def udpPacketKey(packet):
    return packet.udp_packet.m_header.m_sessionTime

def poseSequenceLabelKey(label):
    return label.car_pose.session_time

parser = argparse.ArgumentParser()
parser.add_argument("db_path", help="Path to root directory of DB",  type=str)
parser.add_argument("lookahead_indices", help="Number of indices to look forward for each label",  type=int)
parser.add_argument("--sample_indices", help="Number of indices to sample the spline for on the region specified by lookahead_indices",  type=int, default=60)
parser.add_argument("--max_distance", help="Ignore other agents further than this many meters away",  type=float, default=60)
parser.add_argument("--spline_degree", help="Spline degree to fit",  type=int, default=3)
parser.add_argument("--debug", help="Display debug plots", action="store_true", required=False)
parser.add_argument("--output_dir", help="Output directory for the labels. relative to the database images folder",  default="multi_agent_labels", required=False)
parser.add_argument("--override", help="Always delete existing directories without prompting the user", action="store_true")

args = parser.parse_args()
lookahead_indices = args.lookahead_indices
sample_indices = args.sample_indices
spline_degree = args.spline_degree
root_dir = args.db_path
debug = args.debug
max_distance = args.max_distance
override = args.override
motion_data_folder = os.path.join(root_dir,"udp_data","motion_packets")
image_folder = os.path.join(root_dir,"images")
session_folder = os.path.join(root_dir,"udp_data","session_packets")

with open(os.path.join(root_dir,"f1_dataset_config.yaml"),"r") as f:
    dset_config = yaml.load(f)#, Loader=yaml.SafeLoader)

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

spectating_flags = [bool(packet.udp_packet.m_isSpectating) for packet in session_packets]
spectating = any(spectating_flags)
spectator_car_indices = [int(packet.udp_packet.m_spectatorCarIndex) for packet in session_packets]
print(spectating_flags)
print(spectator_car_indices)
print(spectating)
spectator_car_indices_set = set(spectator_car_indices)
if spectating:
    if len(spectator_car_indices_set)>1:
        raise ValueError("Spectated datasets are only supported if you only spectate 1 car the entire time.")
    else:
        ego_vehicle_index = spectator_car_indices[0]
else:
    ego_vehicle_index = None

image_tags = getAllImageFilePackets(image_folder, use_json)
motion_packets = getAllMotionPackets(motion_data_folder, use_json)
image_tags, image_session_timestamps, motion_packets, slope, intercept = pose_utils.registerImagesToMotiondata(motion_packets,image_tags)
image_system_timestamps = np.array([data.timestamp/1000.0 for data in image_tags])

motion_packet_session_times = np.array([packet.udp_packet.m_header.m_sessionTime for packet in motion_packets])
motion_packet_system_times = np.array([packet.timestamp/1000.0 for packet in motion_packets])



#extract ego vehicle pose information from motion packets
ego_vehicle_poses = [extractPose(packet.udp_packet, car_index=ego_vehicle_index) for packet in motion_packets]
ego_vehicle_positions = np.array([pose[0] for pose in ego_vehicle_poses])
ego_vehicle_position_diffs = np.diff(ego_vehicle_positions, axis=0)
ego_vehicle_position_diff_norms = la.norm(ego_vehicle_position_diffs, axis=1)
ego_vehicle_velocities = np.array([extractVelocity(packet.udp_packet, car_index=ego_vehicle_index) for packet in motion_packets])

ego_vehicle_quaternions = np.array([pose[1] for pose in ego_vehicle_poses])
ego_vehicle_rotations = Rot.from_quat(ego_vehicle_quaternions)
print(motion_packet_session_times.shape)
print(ego_vehicle_positions.shape)
ego_vehicle_position_interpolant = scipy.interpolate.make_interp_spline(motion_packet_session_times, ego_vehicle_positions, k=spline_degree)
ego_vehicle_rotation_interpolant = RotSpline(motion_packet_session_times, ego_vehicle_rotations)
ego_vehicle_velocity_interpolant = scipy.interpolate.make_interp_spline(motion_packet_session_times, ego_vehicle_velocities, k=spline_degree)


# for i in range(0,20):
#     if

try:
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure("F1 Session Time vs System Time")
  skipn=100
  plt.scatter(motion_packet_system_times[::skipn], motion_packet_session_times[::skipn], label='measured data',facecolors='none', edgecolors='b', s=20)
  if intercept>=0:
      plotlabel = 'fitted line:\n y=%f*x+%f' % (slope, intercept)
  else:
      plotlabel = 'fitted line:\n y=%f*x-%f' % (slope, -intercept)
  plt.plot(motion_packet_system_times, slope*motion_packet_system_times + intercept, label=plotlabel, color='black')
  plt.title("F1 Session Time vs System Time", fontsize=20)
  plt.xlabel("OS-tagged System Time", fontsize=20)
  plt.ylabel("F1 Session Time", fontsize=20)
  #plt.rcParams.update({'font.size': 12})
  #plt.rcParams.update({'font.size': 12})
  
  #plt.plot(image_timestamps, label='image tag times')
  fig.legend(loc='center right')#,fontsize=20)

  fig.savefig( os.path.join( output_dir, "datalogger_remap_plot.png" ), bbox_inches='tight')
  fig.savefig( os.path.join( output_dir, "datalogger_remap_plot.eps" ), format='eps', bbox_inches='tight')
  fig.savefig( os.path.join( output_dir, "datalogger_remap_plot.pdf" ), format='pdf', bbox_inches='tight')
  fig.savefig( os.path.join( output_dir, "datalogger_remap_plot.svg" ), format='svg', bbox_inches='tight')



  plt.show()
except KeyboardInterrupt:
  exit(0)
except Exception as e:
  print("Skipping visualiation")
  print(e)


lmdb_dir = os.path.join(output_dir,"lmdb")
if os.path.isdir(lmdb_dir):
    shutil.rmtree(lmdb_dir)
os.makedirs(lmdb_dir)
print("Generating interpolated labels")
config_dict : dict = {"lookahead_indices": lookahead_indices, "regression_slope": slope, "regression_intercept": intercept}

with open(os.path.join(output_dir,'config.yaml'), 'w') as yaml_file:
    yaml.dump(config_dict, yaml_file)#, Dumper=yaml.SafeDumper)
db = deepracing.backend.MultiAgentLabelLMDBWrapper()
db.openDatabase( lmdb_dir, mapsize=int(round(76000*len(image_tags)*1.25)) , max_spare_txns=16, readonly=False, lock=True )

for idx in tqdm(range(len(image_tags))):
    label_tag = MultiAgentLabel_pb2.MultiAgentLabel()
    try:
        label_tag.image_tag.CopyFrom(image_tags[idx])
        tlabel = image_session_timestamps[idx]
        lowerbound = bisect.bisect_left(motion_packet_session_times,tlabel)
        if lowerbound <= round(1.25*lookahead_indices,None):
            continue
        upperbound = lowerbound+lookahead_indices
        if(upperbound >= ( len(motion_packets) - round(1.25*lookahead_indices,None) ) ):
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
        
        dt = motion_packet_session_times[upperbound] - motion_packet_session_times[lowerbound]
        tmax = tlabel + dt
        if ( tmax >= motion_packet_session_times[-1] ) or ( tmax >= image_session_timestamps[-1] ):
            continue
        tsamp = np.linspace(tlabel, tmax, sample_indices)

        PoseMatrixEgo = np.eye(4)
        PoseMatrixEgo[0:3,0:3] = ego_vehicle_rotation.as_matrix()
        PoseMatrixEgo[0:3,3] = ego_vehicle_position
        #PoseMatrixEgoInverse = pose_utils.inverseTransform(PoseMatrixEgo)
        PoseMatrixEgoInverse = np.linalg.inv(PoseMatrixEgo)

        future_ego_positions = ego_vehicle_position_interpolant(tsamp).astype(np.float64)
        future_ego_rotations = ego_vehicle_rotation_interpolant(tsamp)
        future_ego_poses = pose_utils.toHomogenousTransformArray(future_ego_positions, future_ego_rotations.as_quat())
        future_ego_poses = np.matmul(PoseMatrixEgoInverse,future_ego_poses)

        future_ego_linear_velocities = ego_vehicle_velocity_interpolant(tsamp).transpose()
        future_ego_linear_velocities = np.matmul(PoseMatrixEgoInverse[0:3,0:3],future_ego_linear_velocities)

        future_ego_angular_velocities = ego_vehicle_rotation_interpolant(tsamp,1).transpose()
        future_ego_angular_velocities = np.matmul(PoseMatrixEgoInverse[0:3,0:3],future_ego_angular_velocities)

        for i in range(future_ego_poses.shape[0]):
            pose = label_tag.ego_agent_trajectory.poses.add()
            linear_vel = label_tag.ego_agent_trajectory.linear_velocities.add()
            angular_vel = label_tag.ego_agent_trajectory.angular_velocities.add()

            pose.translation.CopyFrom(proto_utils.vectorFromNumpy(future_ego_poses[i,0:3,3]))
            pose.rotation.CopyFrom(proto_utils.quaternionFromScipy(Rot.from_matrix(future_ego_poses[i,0:3,0:3])))
            pose.session_time = tsamp[i]
            pose.frame = FrameId_pb2.LOCAL

            linear_vel.vector.CopyFrom(proto_utils.vectorFromNumpy(future_ego_linear_velocities[:,i]))
            linear_vel.session_time = tsamp[i]
            linear_vel.frame = FrameId_pb2.LOCAL

            angular_vel.vector.CopyFrom(proto_utils.vectorFromNumpy(future_ego_angular_velocities[:,i]))
            angular_vel.session_time = tsamp[i]
            angular_vel.frame = FrameId_pb2.LOCAL


        motion_packets_local = motion_packets[lowerbound-25:upperbound+25]
        session_times_local = np.array([packet.udp_packet.m_header.m_sessionTime for packet in motion_packets_local])
        for i in range(0,20):
            if any([p.udp_packet.m_header.m_playerCarIndex==i for p in motion_packets_local]):
               # print("ignoring index %d. That is the index for the ego vehicle" %(i,))
                continue
            vehicle_poses = [extractPose(packet.udp_packet, car_index=i) for packet in motion_packets_local]
            vehicle_positions = np.array([pose[0] for pose in vehicle_poses])
            vehicle_position_diffs = np.diff(vehicle_positions, axis=0)
            vehicle_position_diff_norms = la.norm(vehicle_position_diffs, axis=1)
            if (np.sum(vehicle_positions)<1.0):# or (np.sum(vehicle_position_diff_norms)<1.0):
                continue

            vehicle_velocities = np.array([extractVelocity(packet.udp_packet, car_index=i) for packet in motion_packets_local])
            vehicle_quaternions = np.array([pose[1] for pose in vehicle_poses])
            if np.isnan(np.sum(vehicle_quaternions)):
                continue
            vehicle_rotations = Rot.from_quat(vehicle_quaternions)
            
            maxnormdiff = np.max(vehicle_position_diff_norms)
            if maxnormdiff>3.5:
                raise ValueError("Saw vehicle %i jump by a distance of %f for image %d. ignoring image %d" % (i, maxnormdiff, idx, idx))
            #print("Max local diff for vehicle %d: %f" % (i, np.max(vehicle_position_diff_norms)) )
            vehicle_position_interpolant = scipy.interpolate.make_interp_spline(session_times_local, vehicle_positions)
            vehicle_rotation_interpolant = RotSpline(session_times_local, vehicle_rotations)
            vehicle_velocity_interpolant = scipy.interpolate.make_interp_spline(session_times_local, vehicle_velocities)

            future_vehicle_positions = vehicle_position_interpolant(tsamp).astype(np.float64)
            future_vehicle_rotations = vehicle_rotation_interpolant(tsamp)
            future_vehicle_poses = pose_utils.toHomogenousTransformArray(future_vehicle_positions, future_vehicle_rotations.as_quat())
            future_vehicle_poses = np.matmul(PoseMatrixEgoInverse,future_vehicle_poses)

            future_vehicle_linear_velocities = vehicle_velocity_interpolant(tsamp).transpose()
            future_vehicle_linear_velocities = np.matmul(PoseMatrixEgoInverse[0:3,0:3], future_vehicle_linear_velocities )

            future_vehicle_angular_velocities = vehicle_rotation_interpolant(tsamp,1).transpose()
            future_vehicle_angular_velocities = np.matmul(PoseMatrixEgoInverse[0:3,0:3], future_vehicle_angular_velocities )

            cvp_x, cvp_y, cvp_z = future_vehicle_poses[0,0,3], future_vehicle_poses[0,1,3], future_vehicle_poses[0,2,3]
            if cvp_z>-0.75 and cvp_z<80.0 and np.abs(cvp_x)<40:
               # print("Got what looks like a match for vehicle %d. It should appear in image %d at position %s" % (i, idx, str(current_vehicle_position_wrt_ego)))
                trajectory = label_tag.other_agent_trajectories.add()
                for j in range(future_vehicle_positions.shape[0]):
                    pose = trajectory.poses.add()
                    pose.translation.CopyFrom(proto_utils.vectorFromNumpy(future_vehicle_poses[j,0:3,3]))
                    pose.rotation.CopyFrom(proto_utils.quaternionFromScipy(Rot.from_matrix(future_vehicle_poses[j,0:3,0:3])))
                    pose.session_time = tsamp[j]
                    pose.frame = FrameId_pb2.LOCAL

                    linear_vel = trajectory.linear_velocities.add()
                    linear_vel.vector.CopyFrom(proto_utils.vectorFromNumpy(future_vehicle_linear_velocities[:,j]))
                    linear_vel.session_time = tsamp[j]
                    linear_vel.frame = FrameId_pb2.LOCAL

                    angular_vel = trajectory.angular_velocities.add()
                    angular_vel.vector.CopyFrom(proto_utils.vectorFromNumpy(future_vehicle_angular_velocities[:,j]))
                    angular_vel.session_time = tsamp[j]
                    angular_vel.frame = FrameId_pb2.LOCAL
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        print("Could not generate label for %s" %(label_tag.image_tag.image_file))
        print("Exception message: %s"%(str(e)))
        continue    
        #raise e      
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
    if debug and idx%30==0:
        fig1 = plt.subplot(1, 2, 1)
        imcv = cv2.imread(os.path.join(image_folder, label_tag.image_tag.image_file), cv2.IMREAD_UNCHANGED)
        plt.imshow(cv2.cvtColor(imcv, cv2.COLOR_BGR2RGB))

        fig2 = plt.subplot(1, 2, 2)
        ego_label_positions = np.array([[pose.translation.x,pose.translation.y,pose.translation.z]  for pose in label_tag.ego_agent_trajectory.poses ])
        print("Ego position 0: " + str(ego_label_positions[0]))
        plt.plot(-ego_label_positions[:,0], ego_label_positions[:,2], label="Ego Vehicle Trajectory", c="b")
        plt.legend()
        for j in range(len(label_tag.other_agent_trajectories)):
            agent_trajectory = label_tag.other_agent_trajectories[j]
            label_positions = np.array([[pose.translation.x,pose.translation.y,pose.translation.z]  for pose in agent_trajectory.poses ])
            plt.plot(-label_positions[:,0], label_positions[:,2])#, label="Ego Vehicle Trajectory", c="b")
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





