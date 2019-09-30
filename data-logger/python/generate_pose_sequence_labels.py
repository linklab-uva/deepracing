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
import Spline2DParams_pb2
def imageDataKey(data):
    return data.timestamp
def udpPacketKey(packet):
    return packet.udp_packet.m_header.m_sessionTime

def poseSequenceLabelKey(label):
    return label.car_pose.session_time

parser = argparse.ArgumentParser()
parser.add_argument("db_path", help="Path to root directory of DB",  type=str)
parser.add_argument("lookahead_indices", help="Time (in seconds) to look foward. Each label will have <num_label_poses> spanning <lookahead_time> seconds into the future",  type=int)
angvelhelp = "Use the angular velocities given in the udp packets. THESE ARE ONLY PROVIDED FOR A PLAYER CAR. IF THE " +\
    " DATASET WAS TAKEN ON SPECTATOR MODE, THE ANGULAR VELOCITY VALUES WILL BE GARBAGE."
parser.add_argument("--splineXmin", help="Min Scale factor for computing the X-component of spline fit",  type=float, default=0.0)
parser.add_argument("--splineXmax", help="Max Scale factor for computing the X-component of spline fit",  type=float, default=1.0)
parser.add_argument("--splineZmin", help="Min Scale factor for computing the Z-component of spline fit",  type=float, default=0.0)
parser.add_argument("--splineZmax", help="Max Scale factor for computing the Z-component of spline fit",  type=float, default=1.0)
parser.add_argument("--splineK", help="Spline degree to fit",  type=int, default=3)
parser.add_argument("--use_given_angular_velocities", help=angvelhelp, action="store_true", required=False)
parser.add_argument("--debug", help="Display debug plots", action="store_true", required=False)
parser.add_argument("--json", help="Assume dataset files are in JSON rather than binary .pb files.",  action="store_true", required=False)
parser.add_argument("--downsample_factor", help="What factor to downsample the measured trajectories by", type=int,  default=1)
parser.add_argument("--output_dir", help="Output directory for the labels. relative to the database images folder",  default="pose_sequence_labels", required=False)

args = parser.parse_args()
downsample_factor = args.downsample_factor
splxmin = args.splineXmin
splxmax = args.splineXmax
splzmin = args.splineZmin
splzmax = args.splineZmax
splK = args.splineK
lookahead_indices = args.lookahead_indices
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
print(len(motion_packets))
print(len(session_times))
print(len(system_times))
print(len(angular_velocities))
print(len(poses))
print(len(positions))
print(len(velocities))
print(len(quaternions))
print()
slope_session_time_fit, intercept_session_time_fit, _, _, _ = scipy.stats.linregress(np.linspace(1,session_times.shape[0],session_times.shape[0]), session_times)
print("Slope and intercept of raw session times: [%f,%f]" %(slope_session_time_fit, intercept_session_time_fit))

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(system_times, session_times)

image_session_timestamps = slope*image_timestamps + intercept
print("Range of image session times before clipping: [%f,%f]" %(image_session_timestamps[0], image_session_timestamps[-1]))

Iclip = (image_session_timestamps>np.min(session_times)) * (image_session_timestamps<np.max(session_times))
image_tags = [image_tags[i] for i in range(len(image_session_timestamps)) if Iclip[i]]
image_session_timestamps = image_session_timestamps[Iclip]
#and image_session_timestamps<np.max(session_timestamps)
print("Range of image session times after clipping: [%f,%f]" %(image_session_timestamps[0], image_session_timestamps[-1]))


position_interpolant = scipy.interpolate.make_interp_spline(session_times, positions)
#velocity_interpolant = position_interpolant.derivative()
velocity_interpolant = scipy.interpolate.make_interp_spline(session_times, velocities)
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
lmdb_dir = os.path.join(output_dir,"lmdb")
if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
if os.path.isdir(lmdb_dir):
    shutil.rmtree(lmdb_dir)
os.makedirs(lmdb_dir)
print("Generating interpolated labels")
config_dict = {"lookahead_indices": lookahead_indices, "downsample_factor": downsample_factor}
with open(os.path.join(output_dir,'config.yaml'), 'w') as yaml_file:
    yaml.dump(config_dict, yaml_file, Dumper=yaml.SafeDumper)
db = deepracing.backend.PoseSequenceLabelLMDBWrapper()
db.readDatabase(lmdb_dir, mapsize=3e9, max_spare_txns=16, readonly=False )

for idx in tqdm(range(len(image_tags))):
    try:
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
        lowerbound = bisect.bisect_left(session_times,t_interp)
        upperbound = lowerbound+lookahead_indices
        interpolants_start = lowerbound
        if(interpolants_start<10):
            continue
        interpolants_end = upperbound
        if interpolants_end>=(len(session_times)-round(1.5*lookahead_indices)):
            continue

        subsequent_positions = positions[lowerbound:upperbound]
        subsequent_quaternions = quaternions[lowerbound:upperbound]
        subsequent_velocities = velocities[lowerbound:upperbound]
        subsequent_angular_velocities = angular_velocities[lowerbound:upperbound]
        subsequent_times = session_times[lowerbound:upperbound]

        
        
        carposition_global = interpolated_positions[idx]

        carvelocity_global = interpolated_velocities[idx]

        carquat_global = interpolated_quaternions[idx]

        carangvelocity_global = interpolated_angular_velocities[idx]

        subsequent_positions_local, subsequent_quaternions_local = deepracing.pose_utils.toLocalCoordinatesPose((carposition_global, carquat_global), subsequent_positions, subsequent_quaternions)
        subsequent_velocities_local = deepracing.pose_utils.toLocalCoordinatesVector((carposition_global, carquat_global), subsequent_velocities)
        subsequent_angular_velocities_local = deepracing.pose_utils.toLocalCoordinatesVector((carposition_global, carquat_global), subsequent_angular_velocities)

        position_spline_ordinates = subsequent_positions_local[::downsample_factor,[0,2]].copy()
        tspline = subsequent_times[::downsample_factor].copy()
        tspline = (tspline-tspline[0])/(tspline[-1]-tspline[0])
        #tspline = np.linspace(0.0,1.0,position_spline_ordinates.shape[0])
        numpoints = position_spline_ordinates.shape[0]
        knots = np.hstack((np.zeros(splK+1),np.linspace((splK-1)/(numpoints-splK+1),(numpoints-splK-1)/(numpoints-splK+1),numpoints-splK-1),np.ones(splK+1)))
        #knots = None
        position_spline_ordinates[:,0] = (position_spline_ordinates[:,0] - splxmin)/(splxmax - splxmin)
        position_spline_ordinates[:,1] = (position_spline_ordinates[:,1] - splzmin)/(splzmax - splzmin)
        position_spline = scipy.interpolate.make_interp_spline(tspline,position_spline_ordinates,k=splK,t=knots)
        position_spline_pb = deepracing.protobuf_utils.splineSciPyToPB(position_spline,tspline[0],tspline[-1],splxmin,splxmax,splzmin,splzmax)
        label_tag.position_spline.CopyFrom(position_spline_pb)


        velocity_spline_ordinates = subsequent_velocities_local[::downsample_factor,[0,2]].copy()
        velocity_spline_ordinates[:,0] = (velocity_spline_ordinates[:,0] - splxmin)/(splxmax - splxmin)
        velocity_spline_ordinates[:,1] = (velocity_spline_ordinates[:,1] - splzmin)/(splzmax - splzmin)
        velocity_spline = scipy.interpolate.make_interp_spline(tspline,velocity_spline_ordinates,k=splK,t=knots)
        velocity_spline_pb = deepracing.protobuf_utils.splineSciPyToPB(velocity_spline,tspline[0],tspline[-1],splxmin,splxmax,splzmin,splzmax)
        label_tag.velocity_spline.CopyFrom(velocity_spline_pb)


        
        

            
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



        if args.debug and (idx%30)==0:
            print("Final point in undecimated list: " + str(position_spline_ordinates[-1]))
            print("Final point in decimated list: " + str(subsequent_positions_local[-1,[0,2]]))
            print(google.protobuf.json_format.MessageToJson(label_tag.position_spline,including_default_value_fields=True))
            print(subsequent_times.shape)
            print("Mean diff of time vector: %f" %(np.mean(np.diff(subsequent_times))))
            tsamp = np.linspace(0.0,1.0,16)
            position_spline_rebuilt = deepracing.protobuf_utils.splinePBToSciPy(label_tag.position_spline)
            position_resamp = position_spline_rebuilt(tsamp)
            spline_derivative = position_spline_rebuilt.derivative()
            velocity_resamp = spline_derivative(tsamp)
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(subsequent_positions_local[:,0],subsequent_positions_local[:,2], 'bo')
            ax.plot(label_tag.position_spline.Xmin + position_resamp[:,0]*(label_tag.position_spline.Xmax - label_tag.position_spline.Xmin),\
                    label_tag.position_spline.Zmin + position_resamp[:,1]*(label_tag.position_spline.Zmax - label_tag.position_spline.Zmin),\
                    'ro')
            ax.quiver(subsequent_positions_local[:,0],subsequent_positions_local[:,2],subsequent_velocities_local[:,0],subsequent_velocities_local[:,2])
            #ax.quiver(position_resamp[:,0],position_resamp[:,1],velocity_resamp[:,0],velocity_resamp[:,1])
            # figvel = plt.figure()
            # axvel = figvel.add_subplot()
            # axvel.plot(subsequent_velocities_local[:,0],subsequent_velocities_local[:,2], 'bo')
            # axvel.plot(label_tag.position_spline.Xmin + velocity_resamp[:,0]*(label_tag.position_spline.Xmax - label_tag.position_spline.Xmin),\
            #         label_tag.position_spline.Zmin + velocity_resamp[:,1]*(label_tag.position_spline.Zmax - label_tag.position_spline.Zmin),\
            #         'ro')
            plt.show()
        #print()
        #print()
        #print(carposition_global)
        #print(carpose_global)
        #print()
        for j in range(len(subsequent_positions_local)-1):
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

            labeltime = subsequent_times[j]
            pose_forward_pb.session_time = labeltime
            velocity_forward_pb.session_time = labeltime
            angular_velocity_forward_pb.session_time = labeltime

            newpose = label_tag.subsequent_poses.add()
            newpose.CopyFrom(pose_forward_pb)

            newvel = label_tag.subsequent_linear_velocities.add()
            newvel.CopyFrom(velocity_forward_pb)

            newangvel = label_tag.subsequent_angular_velocities.add()
            newangvel.CopyFrom(angular_velocity_forward_pb)
    except Exception as e:
        print("Could not generate label for %s" %(label_tag.image_tag.image_file))
        print("Exception message: %s"%(str(e)))
        continue          
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
print("Loading database labels.")
db_keys = db.getKeys()
label_pb_tags = []
for i,key in tqdm(enumerate(db_keys), total=len(db_keys)):
    #print(key)
    label_pb_tags.append(db.getPoseSequenceLabel(key))
    if(not (label_pb_tags[-1].image_tag.image_file == db_keys[i]+".jpg")):
        raise AttributeError("Mismatch between database key: %s and associated image file: %s" %(db_keys[i], label_pb_tags.image_tag.image_file))
sorted_indices = np.argsort( np.array([lbl.car_pose.session_time for lbl in label_pb_tags]) )
#tagscopy = label_pb_tags.copy()
label_pb_tags_sorted = [label_pb_tags[sorted_indices[i]] for i in range(len(sorted_indices))]
sorted_keys = []
print("Checking for invalid keys.")
for packet in tqdm(label_pb_tags_sorted):
    #print(key)
    key = os.path.splitext(packet.image_tag.image_file)[0]
    try:
        lbl = db.getPoseSequenceLabel(key)
        sorted_keys.append(key)
    except:
        print("Skipping bad key: %s" %(key))
        continue
key_file = os.path.join(args.db_path,"goodkeys.txt")
with open(key_file, 'w') as filehandle:
    filehandle.writelines("%s\n" % key for key in sorted_keys)    





