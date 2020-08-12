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
import json 
from scipy.spatial import KDTree as KDTree


def imageDataKey(data):
    return data.timestamp

def poseSequenceLabelKey(label):
    return label.car_pose.session_time

parser = argparse.ArgumentParser()
parser.add_argument("db_path", help="Path to root directory of DB",  type=str)
parser.add_argument("raceline", help="Path to the json file containing the raceline",  type=str)
parser.add_argument("lookahead_distance", help="Distance to look ahead on the raceline. This means the labels can have variable length",  type=float)
parser.add_argument("--debug", help="Display debug plots", action="store_true", required=False)
parser.add_argument("--output_dir", help="Output directory for the labels. relative to the database images folder",  default="raceline_labels", required=False)

args = parser.parse_args()
lookahead_distance = args.lookahead_distance
root_dir = args.db_path
racelinepath = args.raceline
with open(os.path.join(root_dir,"f1_dataset_config.yaml"),"r") as f:
    config = yaml.load(f,Loader=yaml.SafeLoader)
    use_json = config["use_json"]


motion_data_folder = os.path.join(root_dir,"udp_data","motion_packets")
image_folder = os.path.join(root_dir,"images")
session_folder = os.path.join(root_dir,"udp_data","session_packets")
session_packets = getAllSessionPackets(session_folder,use_json)
output_dir = os.path.join(root_dir, args.output_dir)
if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
with open(racelinepath,"r") as f:
    racelinedict = json.load(f)
racelinedist = np.array(racelinedict["dist"])
racelineaug = np.row_stack((racelinedict["x"],racelinedict["y"],racelinedict["z"],np.ones_like(racelinedict["x"])))

print(racelineaug)
spectating_flags = [bool(packet.udp_packet.m_isSpectating) for packet in session_packets]
spectating = any(spectating_flags)
car_indices = [int(packet.udp_packet.m_spectatorCarIndex) for packet in session_packets]
print(spectating_flags)
print(car_indices)
print(spectating)
car_indices_set = set(car_indices)
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
image_tags = [tag for tag in image_tags if tag.timestamp/1000.0<maxudptime]
image_tags = sorted(image_tags, key = imageDataKey)
image_timestamps = np.array([data.timestamp/1000.0 for data in image_tags])


first_image_time = image_timestamps[0]
print(first_image_time)
Imin = system_times>(first_image_time + 1.0)
firstIndex = np.argmax(Imin)

motion_packets = motion_packets[firstIndex:]
motion_packets = sorted(motion_packets, key=deepracing.timestampedUdpPacketKey)
session_times = np.array([packet.udp_packet.m_header.m_sessionTime for packet in motion_packets])
unique_session_times, unique_session_time_indices = np.unique(session_times, return_index=True)
motion_packets = [motion_packets[i] for i in unique_session_time_indices]
motion_packets = sorted(motion_packets, key=deepracing.timestampedUdpPacketKey)
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

slope_session_time_fit, intercept_session_time_fit, rvalue, pvalue, stderr = scipy.stats.linregress(np.linspace(1,session_times.shape[0],session_times.shape[0]), session_times)
print("Slope and intercept of raw session times: [%f,%f]" %(slope_session_time_fit, intercept_session_time_fit))

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(system_times, session_times)
print("Slope and intercept of session time vs system time: [%f,%f]" %(slope, intercept))
print( "r value of session time vs system time: %f" % ( rvalue ) )
print( "r^2 value of session time vs system time: %f" % ( rvalue**2 ) )

image_session_timestamps = slope*image_timestamps + intercept
print("Range of image session times before clipping: [%f,%f]" %(image_session_timestamps[0], image_session_timestamps[-1]))

Iclip = (image_session_timestamps>np.min(session_times)) * (image_session_timestamps<np.max(session_times))
image_tags = [image_tags[i] for i in range(len(image_session_timestamps)) if Iclip[i]]
image_session_timestamps = image_session_timestamps[Iclip]
print("Range of image session times after clipping: [%f,%f]" %(image_session_timestamps[0], image_session_timestamps[-1]))


position_interpolant = scipy.interpolate.make_interp_spline(session_times, positions)
rotation_interpolant = RotSpline(session_times,rotations)
velocity_interpolant = scipy.interpolate.make_interp_spline(session_times, velocities)


interpolated_positions = position_interpolant(image_session_timestamps)
interpolated_velocities = velocity_interpolant(image_session_timestamps)
interpolated_quaternions = rotation_interpolant(image_session_timestamps)
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
  fig = plt.figure("F1 Session Time vs System Time")
  skipn=100
  plt.scatter(system_times[::skipn], session_times[::skipn], label='measured data',facecolors='none', edgecolors='b', s=8)
  if intercept>=0:
      plotlabel = 'fitted line:\n y=%f*x+%f' % (slope, intercept)
  else:
      plotlabel = 'fitted line:\n y=%f*x-%f' % (slope, -intercept)
  plt.plot(system_times, slope*system_times + intercept, label=plotlabel,color='black')
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
  #text = input("Could not import matplotlib, skipping visualization. Enter anything to continue.")
#scipy.interpolate.interp1d
lmdb_dir = os.path.join(output_dir,"lmdb")
if os.path.isdir(lmdb_dir):
    shutil.rmtree(lmdb_dir)
os.makedirs(lmdb_dir)
print("Generating raceline labels")
config_dict = {"lookahead_distance": lookahead_distance}
with open(os.path.join(output_dir,'config.yaml'), 'w') as yaml_file:
    yaml.dump(config_dict, yaml_file, Dumper=yaml.SafeDumper)
#exit(0)
db = deepracing.backend.PoseSequenceLabelLMDBWrapper()
db.readDatabase( lmdb_dir, mapsize=int(round(9996*len(image_tags)*1.25)), max_spare_txns=16, readonly=False )
debug = False
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
        
        
        car_position = position_interpolant(label_tag.car_pose.session_time)
        car_rotation = rotation_interpolant(label_tag.car_pose.session_time)
        car_velocity = velocity_interpolant(label_tag.car_velocity.session_time)
        car_angular_velocity = rotation_interpolant(label_tag.car_angular_velocity.session_time,order=1)

        car_affine_pose = np.eye(4)
        car_affine_pose[0:3,0:3] = car_rotation.as_matrix()
        car_affine_pose[0:3,3] = car_position
        car_affine_pose_inv = la.inv(car_affine_pose)
        raceline_local = np.matmul(car_affine_pose_inv,racelineaug)[0:3,:].transpose()
        Izpos = raceline_local[:,2]>=0.0
        raceline_geodesic_distances = racelinedist[Izpos]
        raceline_local = raceline_local[Izpos]
        raceline_distances = np.linalg.norm(raceline_local,axis=1)
        I1 = np.argmin(raceline_distances)
        d1 = raceline_geodesic_distances[I1]
        d2 = d1 + lookahead_distance
        I2 = bisect.bisect_left(raceline_geodesic_distances, d2)
        if d2 < raceline_geodesic_distances[-1]:
          #  print("Normal label.")# I1: %d. I2: %d" % (I1, I2))
           # print(raceline_local[I1:I2])
            local_points = raceline_local[I1:I2].copy()
            local_geodesic_distances = raceline_geodesic_distances[I1:I2].copy()
        else:
            I2 = bisect.bisect_left(raceline_geodesic_distances,  d2 - raceline_geodesic_distances[-1]) 
            local_points = np.vstack( ( raceline_local[I1:],  raceline_local[0:I2])  )
            local_geodesic_distances = np.hstack( ( raceline_geodesic_distances[I1:],  raceline_geodesic_distances[0:I2] + raceline_geodesic_distances[-1] ) )
            
            if debug:# and (idx%10==0):
                f, (imax, plotax) = plt.subplots(nrows=1 , ncols=2)
                im = cv2.cvtColor(cv2.imread(os.path.join(image_folder,label_tag.image_tag.image_file)), cv2.COLOR_BGR2RGB)
                imax.imshow(im)
                plotax.plot(-local_points[:,0], local_points[:,2])
                try:
                    plt.show()
                except Exception as e:
                    print("Skipping visualization")
                  #  plt.close('all')
            #continue

        label_tag.car_pose.translation.x = car_position[0]
        label_tag.car_pose.translation.y = car_position[1]
        label_tag.car_pose.translation.z = car_position[2]
        car_quaternion = car_rotation.as_quat()
        label_tag.car_pose.rotation.x = car_quaternion[0]
        label_tag.car_pose.rotation.y = car_quaternion[1]
        label_tag.car_pose.rotation.z = car_quaternion[2]
        label_tag.car_pose.rotation.w = car_quaternion[3]

        label_tag.car_velocity.vector.x = car_velocity[0]
        label_tag.car_velocity.vector.y = car_velocity[1]
        label_tag.car_velocity.vector.z = car_velocity[2]

        label_tag.car_angular_velocity.vector.x = car_angular_velocity[0]
        label_tag.car_angular_velocity.vector.y = car_angular_velocity[1]
        label_tag.car_angular_velocity.vector.z = car_angular_velocity[2]
      #  print("Writing a label with %d points" % (local_points.shape[0],) )
        for j in range(local_points.shape[0]):
            newpose = label_tag.subsequent_poses.add()
            newpose.translation.x = local_points[j,0]
            newpose.translation.y = local_points[j,1]
            newpose.translation.z = local_points[j,2]
            newpose.rotation.x = 0.0
            newpose.rotation.y = 0.0
            newpose.rotation.z = 0.0
            newpose.rotation.w = 1.0
            newpose.session_time = j/(local_points.shape[0]-1)
            newpose.frame = FrameId_pb2.LOCAL
            
            newvel = label_tag.subsequent_linear_velocities.add()
            newvel.vector.x = 0.0
            newvel.vector.y = 0.0
            newvel.vector.z = 0.0
            newvel.session_time = j/(local_points.shape[0]-1)
            newvel.frame = FrameId_pb2.LOCAL

            newangvel = label_tag.subsequent_angular_velocities.add()
            newangvel.vector.x = 0.0
            newangvel.vector.y = 0.0
            newangvel.vector.z = 0.0
            newangvel.session_time = j/(local_points.shape[0]-1)
            newangvel.frame = FrameId_pb2.LOCAL





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





