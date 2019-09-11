import numpy as np
import numpy.linalg as la
import quaternion
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
from deepracing.pose_utils import getAllImageFilePackets, getAllMotionPackets
from deepracing.protobuf_utils import getAllSessionPackets
from tqdm import tqdm as tqdm
import yaml
import shutil
def imageDataKey(data):
    return data.timestamp
def udpPacketKey(packet):
    return packet.timestamp

parser = argparse.ArgumentParser()
parser.add_argument("db_path", help="Path to root directory of DB",  type=str)
parser.add_argument("num_label_poses", help="Number of poses to attach to each image",  type=int)
parser.add_argument("lookahead_time", help="Time (in seconds) to look foward. Each label will have <num_label_poses> spanning <lookahead_time> seconds into the future",  type=float)
angvelhelp = "Use the angular velocities given in the udp packets. THESE ARE ONLY PROVIDED FOR A PLAYER CAR. IF THE " +\
    " DATASET WAS TAKEN ON SPECTATOR MODE, THE ANGULAR VELOCITY VALUES WILL BE GARBAGE."
parser.add_argument("--use_given_angular_velocities", help=angvelhelp, action="store_true", required=False)
parser.add_argument("--assume_linear_timescale", help="Assumes the slope between system time and session time is 1.0", action="store_true", required=False)
parser.add_argument("--json", help="Assume dataset files are in JSON rather than binary .pb files.",  action="store_true", required=False)
parser.add_argument("--output_dir", help="Output directory for the labels. relative to the database images folder",  default="pose_sequence_labels", required=False)
parser.add_argument("--lmdb_dir", help="Output directory for the output lmdb. relative to the database images folder",  default="pose_sequence_label_lmdb", required=False)

args = parser.parse_args()
num_label_poses = args.num_label_poses
lookahead_time = args.lookahead_time
motion_data_folder = os.path.join(args.db_path,"udp_data","motion_packets")
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
motion_packets = deepracing.pose_utils.getAllMotionPackets(motion_data_folder, args.json)
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

poses = [deepracing.pose_utils.extractPose(packet.udp_packet, car_index=car_index) for packet in motion_packets]
velocities = np.array([deepracing.pose_utils.extractVelocity(packet.udp_packet, car_index=car_index) for packet in motion_packets])
positions = np.array([pose[0] for pose in poses])
quaternions = np.array([pose[1] for pose in poses])
if args.use_given_angular_velocities:
    angular_velocities = np.array([deepracing.pose_utils.extractAngularVelocity(packet.udp_packet, car_index=0) for packet in motion_packets])
else:
    angular_velocities = quaternion.angular_velocity(quaternions, session_times)
for i in range(len(quaternions)):
    if quaternions[i].w<0:
        quaternions[i]=-quaternions[i]
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
output_dir = os.path.join(image_folder, args.output_dir)
lmdb_dir = os.path.join(image_folder, args.lmdb_dir)
if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
if os.path.isdir(lmdb_dir):
    shutil.rmtree(lmdb_dir)
os.makedirs(lmdb_dir)
print("Generating interpolated labels")
config_dict = {"num_poses": num_label_poses, "lookahead_time":lookahead_time}
with open(os.path.join(output_dir,'config.yaml'), 'w') as yaml_file:
    yaml.dump(config_dict, yaml_file, Dumper=yaml.SafeDumper)
db = deepracing.backend.PoseSequenceLabelLMDBWrapper()
db.readDatabase(lmdb_dir, mapsize=3e9, max_spare_txns=16, readonly=False )
for idx in tqdm(range(len(image_tags))):
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
    interpolants_start = bisect.bisect_left(session_times,t_interp)
    if(interpolants_start<10):
        continue
    interpolants_start = interpolants_start-2
    t_interp_end = t_interp+lookahead_time
    interpolants_end = bisect.bisect_left(session_times,t_interp_end)
    if(interpolants_end>=len(session_times)-10):
        continue
    interpolants_end = interpolants_end+2
   
    position_interpolant_points = positions[interpolants_start:interpolants_end]
    rotation_interpolants_points = quaternions[interpolants_start:interpolants_end]
    velocity_interpolants_points = velocities[interpolants_start:interpolants_end]
    angular_velocity_interpolants_points = angular_velocities[interpolants_start:interpolants_end]
    interpolant_times = session_times[interpolants_start:interpolants_end]
    
    local_position_interpolant = scipy.interpolate.interp1d(interpolant_times, position_interpolant_points , axis=0, kind='cubic')
    local_velocity_interpolant = scipy.interpolate.interp1d(interpolant_times, velocity_interpolants_points, axis=0, kind='cubic')
    local_angular_velocity_interpolant = scipy.interpolate.interp1d(interpolant_times, angular_velocity_interpolants_points, axis=0, kind='cubic')

    t_eval = np.linspace(t_interp, t_interp_end, num_label_poses)

    
    
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

    carpose_global = deepracing.pose_utils.toHomogenousTransform(carposition_global, carquat_global)
    #yes, I know this is an un-necessary inverse computation. Sue me.
    carposeinverse_global = la.inv(carpose_global)

    subsequent_positions = local_position_interpolant(t_eval)
    subsequent_quaternions = quaternion.squad(rotation_interpolants_points, interpolant_times, t_eval)
    subsequent_velocities = local_velocity_interpolant(t_eval)
    subsequent_angular_velocities = local_angular_velocity_interpolant(t_eval)




    subsequent_positions_local, subsequent_quaternions_local = deepracing.pose_utils.toLocalCoordinatesPose((carposition_global, carquat_global), subsequent_positions, subsequent_quaternions)
    subsequent_velocities_local = deepracing.pose_utils.toLocalCoordinatesVector((carposition_global, carquat_global), subsequent_velocities)
    subsequent_angular_velocities_local = deepracing.pose_utils.toLocalCoordinatesVector((carposition_global, carquat_global), subsequent_angular_velocities)
    #print()
    #print()
    #print(carposition_global)
    #print(carpose_global)
    #print()
    for j in range(num_label_poses):
        # label_tag.
        #print(packet_forward)
        pose_forward_pb = Pose3d_pb2.Pose3d()
        velocity_forward_pb = Vector3dStamped_pb2.Vector3dStamped()
        angular_velocity_forward_pb = Vector3dStamped_pb2.Vector3dStamped()
        pose_forward_pb.frame = FrameId_pb2.LOCAL
        velocity_forward_pb.frame = FrameId_pb2.LOCAL
        angular_velocity_forward_pb.frame = FrameId_pb2.LOCAL

        pose_forward_pb.translation.x = subsequent_positions_local[j,0]
        pose_forward_pb.translation.y = subsequent_positions_local[j,1]
        pose_forward_pb.translation.z = subsequent_positions_local[j,2]
        pose_forward_pb.rotation.x = subsequent_quaternions_local[j].x
        pose_forward_pb.rotation.y = subsequent_quaternions_local[j].y
        pose_forward_pb.rotation.z = subsequent_quaternions_local[j].z
        pose_forward_pb.rotation.w = subsequent_quaternions_local[j].w

        velocity_forward_pb.vector.x = subsequent_velocities_local[j,0]
        velocity_forward_pb.vector.y = subsequent_velocities_local[j,1]
        velocity_forward_pb.vector.z = subsequent_velocities_local[j,2]
        
        angular_velocity_forward_pb.vector.x = subsequent_angular_velocities_local[j,0]
        angular_velocity_forward_pb.vector.y = subsequent_angular_velocities_local[j,1]
        angular_velocity_forward_pb.vector.z = subsequent_angular_velocities_local[j,2]

        pose_forward_pb.session_time = t_eval[j]
        velocity_forward_pb.session_time = t_eval[j]
        angular_velocity_forward_pb.session_time = t_eval[j]

        newpose = label_tag.subsequent_poses.add()
        newpose.CopyFrom(pose_forward_pb)

        newvel = label_tag.subsequent_linear_velocities.add()
        newvel.CopyFrom(velocity_forward_pb)

        newangvel = label_tag.subsequent_angular_velocities.add()
        newangvel.CopyFrom(angular_velocity_forward_pb)

        
    label_tag_JSON = google.protobuf.json_format.MessageToJson(label_tag, including_default_value_fields=True)
    image_file_base = os.path.splitext(os.path.split(label_tag.image_tag.image_file)[1])[0]
    label_tag_file_path = os.path.join(output_dir, image_file_base + "_sequence_label.json")
    f = open(label_tag_file_path,'w')
    f.write(label_tag_JSON)
    f.close()
    label_tag_file_path_binary = os.path.join(output_dir, image_file_base + "_sequence_label.pb")
    f = open(label_tag_file_path_binary,'wb')
    f.write( label_tag.SerializeToString() )
    f.close()
    key = image_file_base
    db.writePoseSequenceLabel( key , label_tag )
    





