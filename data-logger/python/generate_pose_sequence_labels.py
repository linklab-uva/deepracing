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
from deepracing.protobuf_utils import getAllSessionPackets, getAllImageFilePackets, getAllMotionPackets, extractPose, extractVelocity
from tqdm import tqdm as tqdm
import yaml
import shutil
import Spline2DParams_pb2
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
parser.add_argument("lookahead_indices", help="Number of indices to look foward. Each label will have <lookahead_indices> values",  type=int)
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
parser.add_argument("--output_dir", help="Output directory for the labels. relative to the database images folder",  default="pose_sequence_labels", required=False)

args = parser.parse_args()
splxmin = args.splineXmin
splxmax = args.splineXmax
splzmin = args.splineZmin
splzmax = args.splineZmax
splK = args.splineK
lookahead_indices = args.lookahead_indices
root_dir = args.db_path
motion_data_folder = os.path.join(root_dir,"udp_data","motion_packets")
image_folder = os.path.join(root_dir,"images")
session_folder = os.path.join(root_dir,"udp_data","session_packets")
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

image_tags = getAllImageFilePackets(image_folder, args.json)
motion_packets = getAllMotionPackets(motion_data_folder, args.json)
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

poses = [extractPose(packet.udp_packet, car_index=car_index) for packet in motion_packets]
velocities = np.array([extractVelocity(packet.udp_packet, car_index=car_index) for packet in motion_packets])
positions = np.array([pose[0] for pose in poses])
position_diffs = np.diff(positions, axis=0)
position_diff_norms = la.norm(position_diffs, axis=1)
print("Diff norm vector has length %d: " % (len(position_diff_norms)))

quaternions = np.array([pose[1] for pose in poses])
# for i in range(quaternions.shape[0]):
#     if quaternions[i,3]<0:
#         quaternions[i] = - quaternions[i]
rotations = Rot.from_quat(quaternions)
print()
print(len(motion_packets))
print(len(session_times))
print(len(system_times))
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
#rotation_interpolant = Slerp(session_times,Rot.from_quat(quaternions))
rotation_interpolant = RotSpline(session_times,rotations)
velocity_interpolant = scipy.interpolate.make_interp_spline(session_times, velocities)
#accelerations = 0.25*position_interpolant(session_times,nu=2) + 0.75*velocity_interpolant(session_times,nu=1)


interpolated_positions = position_interpolant(image_session_timestamps)
interpolated_velocities = velocity_interpolant(image_session_timestamps)
#interpolated_accelerations = 0.25*position_interpolant(image_session_timestamps,nu=2) + 0.75*velocity_interpolant(image_session_timestamps,nu=1)
interpolated_quaternions = rotation_interpolant(image_session_timestamps)
if args.use_given_angular_velocities:
    angular_velocities = np.array([deepracing.pose_utils.extractAngularVelocity(packet.udp_packet) for packet in motion_packets])
    angular_velocity_interpolant = scipy.interpolate.make_interp_spline(session_times, angular_velocities)
    interpolated_angular_velocities = angular_velocity_interpolant(image_session_timestamps)
else:
    angular_velocities = rotation_interpolant(session_times,order=1)
    interpolated_angular_velocities = rotation_interpolant(image_session_timestamps,order=1)
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
  racelinefig,racelineax = plt.subplots()
  racelineax.scatter( positions[:,0], positions[:,2], facecolors='none', edgecolors='g')
  plt.figure()
  plt.scatter( interpolated_positions[:,0], interpolated_positions[:,2], facecolors='none', edgecolors='b')
  plt.figure()
  plt.plot(session_times[1:],position_diff_norms)
#   plt.plot(session_times,positions[:,2])
#   plt.figure()
#   plt.plot(image_session_timestamps,interpolated_positions[:,0])
#   plt.plot(image_session_timestamps,interpolated_positions[:,2])
  plt.show()
except KeyboardInterrupt:
    exit(0)
except Exception as e:
  print(e)
  text = input("Could not import matplotlib, skipping visualization. Enter anything to continue.")
#scipy.interpolate.interp1d
output_dir = os.path.join(root_dir, args.output_dir)
lmdb_dir = os.path.join(output_dir,"lmdb")
if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
if os.path.isdir(lmdb_dir):
    shutil.rmtree(lmdb_dir)
os.makedirs(lmdb_dir)
print("Generating interpolated labels")
config_dict = {"lookahead_indices": lookahead_indices}
with open(os.path.join(output_dir,'config.yaml'), 'w') as yaml_file:
    yaml.dump(config_dict, yaml_file, Dumper=yaml.SafeDumper)
db = deepracing.backend.PoseSequenceLabelLMDBWrapper()
db.readDatabase( lmdb_dir, mapsize=int(round(9996*len(image_tags)*1.25)), max_spare_txns=16, readonly=False )

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
        interpolants_start = lowerbound-3
        if(interpolants_start<5):
            continue
        interpolants_end = upperbound+3
        if interpolants_end>=(len(session_times)-round(1.25*lookahead_indices)):
            continue
        discontinuity_min = max(0,interpolants_start-10)
        discontinuity_max = min(position_diff_norms.shape[0],interpolants_end+10)
        if(np.max(position_diff_norms[discontinuity_min:discontinuity_max])>=4.0):
            print("Discontinuity in labels, ignoring image: %s" % (label_tag.image_tag.image_file))
            continue
        position_interp_points = positions[interpolants_start:interpolants_end]
        quaternion_interp_points = quaternions[interpolants_start:interpolants_end]
        # velocity_interp_points = velocities[interpolants_start:interpolants_end]
        # session_times_interp = session_times[interpolants_start:interpolants_end].copy()
        # position_spline = scipy.interpolate.make_interp_spline(session_times_interp, position_interp_points)
        # quaternions_spline = RotSpline(session_times_interp, Rot.from_quat(quaternion_interp_points))
        # velocities_spline = scipy.interpolate.make_interp_spline(session_times_interp, velocity_interp_points)
        dT = session_times[upperbound]-session_times[lowerbound]
        teval = np.linspace(t_interp,t_interp+dT,lookahead_indices)

        subsequent_positions = position_interpolant(teval)
        subsequent_quaternions = rotation_interpolant(teval).as_quat()
        subsequent_velocities = velocity_interpolant(teval)
        subsequent_angular_velocities = rotation_interpolant(teval,order=1)
        subsequent_times = teval

        carposition_global = subsequent_positions[0].copy()
        carquat_global = subsequent_quaternions[0].copy()
        carvelocity_global = subsequent_velocities[0].copy()
        carangvelocity_global = subsequent_angular_velocities[0].copy()
        
        pose_global = (carposition_global, carquat_global)
        subsequent_positions_local, subsequent_quaternions_local = deepracing.pose_utils.toLocalCoordinatesPose( pose_global , subsequent_positions, subsequent_quaternions )
        subsequent_velocities_local = deepracing.pose_utils.toLocalCoordinatesVector( pose_global , subsequent_velocities )
        subsequent_angular_velocities_local = deepracing.pose_utils.toLocalCoordinatesVector( pose_global , subsequent_angular_velocities )

        # position_spline_ordinates = subsequent_positions_local[:,[0,2]].copy()
        # #print(position_spline_ordinates.shape)
        # tspline = subsequent_times.copy()
        # tspline = (tspline-tspline[0])/(tspline[-1]-tspline[0])
        # #tspline = np.linspace(0.0,1.0,position_spline_ordinates.shape[0])
        # numpoints = position_spline_ordinates.shape[0]
        # knots = np.hstack((np.zeros(splK+1),np.linspace((splK-1)/(numpoints-splK+1),(numpoints-splK-1)/(numpoints-splK+1),numpoints-splK-1),np.ones(splK+1)))
        # #knots = None
        # position_spline_ordinates[:,0] = (position_spline_ordinates[:,0] - splxmin)/(splxmax - splxmin)
        # position_spline_ordinates[:,1] = (position_spline_ordinates[:,1] - splzmin)/(splzmax - splzmin)
        # position_spline = scipy.interpolate.make_interp_spline(tspline,position_spline_ordinates,k=splK,t=knots)
        # position_spline_pb = deepracing.protobuf_utils.splineSciPyToPB(position_spline,tspline[0],tspline[-1],splxmin,splxmax,splzmin,splzmax)
        # label_tag.position_spline.CopyFrom(position_spline_pb)


        # velocity_spline_ordinates = subsequent_velocities_local[:,[0,2]].copy()
        # velocity_spline_ordinates[:,0] = (velocity_spline_ordinates[:,0] - splxmin)/(splxmax - splxmin)
        # velocity_spline_ordinates[:,1] = (velocity_spline_ordinates[:,1] - splzmin)/(splzmax - splzmin)
        # velocity_spline = scipy.interpolate.make_interp_spline(tspline,velocity_spline_ordinates,k=splK,t=knots)
        # velocity_spline_pb = deepracing.protobuf_utils.splineSciPyToPB(velocity_spline,tspline[0],tspline[-1],splxmin,splxmax,splzmin,splzmax)
        # label_tag.velocity_spline.CopyFrom(velocity_spline_pb)
        label_tag.car_pose.translation.x = carposition_global[0]
        label_tag.car_pose.translation.y = carposition_global[1]
        label_tag.car_pose.translation.z = carposition_global[2]
        label_tag.car_pose.rotation.x = carquat_global[0]
        label_tag.car_pose.rotation.y = carquat_global[1]
        label_tag.car_pose.rotation.z = carquat_global[2]
        label_tag.car_pose.rotation.w = carquat_global[3]

        label_tag.car_velocity.vector.x = carvelocity_global[0]
        label_tag.car_velocity.vector.y = carvelocity_global[1]
        label_tag.car_velocity.vector.z = carvelocity_global[2]
        
        label_tag.car_angular_velocity.vector.x = carangvelocity_global[0]
        label_tag.car_angular_velocity.vector.y = carangvelocity_global[1]
        label_tag.car_angular_velocity.vector.z = carangvelocity_global[2]



        if args.debug and (idx%30)==0:
            
            print("Car position:")
            print(carposition_global)
            print("Car RotationMatrix:")
            print(Rot.from_quat(carquat_global).as_dcm())
            interpolants_local, _ = deepracing.pose_utils.toLocalCoordinatesPose((carposition_global, carquat_global), position_interp_points, quaternion_interp_points)
            figimg = plt.figure()
            aximg = figimg.add_subplot()
            aximg.imshow(cv2.cvtColor(cv2.imread(os.path.join(image_folder,label_tag.image_tag.image_file)), cv2.COLOR_BGR2RGB))
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(subsequent_positions_local[:,0],subsequent_positions_local[:,2], 'bo') 
            ax.plot(interpolants_local[:,0],interpolants_local[:,2], 'go') 

            # fig_global = plt.figure()
            # ax_global = fig_global.add_subplot()
            # ax_global.plot(position_interp_points[:,0],position_interp_points[:,2], 'ro') 

            plt.show()
        #print()
        #print()
        #print(carposition_global)
        #print(carpose_global)
        #print()
        for j in range(len(subsequent_positions_local)):
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
            pose_forward_pb.rotation.x = subsequent_quaternions_local[j,0]
            pose_forward_pb.rotation.y = subsequent_quaternions_local[j,1]
            pose_forward_pb.rotation.z = subsequent_quaternions_local[j,2]
            pose_forward_pb.rotation.w = subsequent_quaternions_local[j,3]

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
        #continue    
        raise e      
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
key_file = os.path.join(root_dir,"goodkeys.txt")
with open(key_file, 'w') as filehandle:
    filehandle.writelines("%s\n" % key for key in sorted_keys)    





