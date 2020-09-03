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
import matplotlib.pyplot as plt
import time
from deepracing import trackNames
import json
def imageDataKey(data):
    return data.timestamp

def poseSequenceLabelKey(label):
    return label.car_pose.session_time

parser = argparse.ArgumentParser()
parser.add_argument("db_path", help="Path to root directory of DB",  type=str)
parser.add_argument("--trackfiledir", help="Path to the directory containing the raceline json files. Default is environment variable F1_TRACK_DIR",  type=str, default=os.environ.get("F1_TRACK_DIR",default=None))
parser.add_argument("--lookahead_distance", help="Distance to look ahead on the raceline. This means the labels can have variable length. Mutually exclusive with --lookahead_indices",  type=float, default=None)
parser.add_argument("--lookahead_indices", help="Number of indices to look ahead in the raceline. Mutually exclusive with --lookahead_distance",  type=int, default=None)
parser.add_argument("--debug", help="Display debug plots", action="store_true", required=False)
parser.add_argument("--output_dir", help="Output directory for the labels. relative to the database images folder",  default="raceline_labels", required=False)
parser.add_argument("--max_speed", help="maximum speed (m/s) to apply to the raceline",  default=90, required=False, type=float)
parser.add_argument("--min_speed", help="minimum speed (m/s) to apply to the raceline",  default=15, required=False, type=float)
parser.add_argument("--max_accel", help="maximum centripetal acceleration (m/s^2) to allow in the raceline",  default=22.5, required=False, type=float)
parser.add_argument("--num_samples", help="Number of points to sample from the racing spline",  default=9000, required=False, type=int)



args = parser.parse_args()
lookahead_distance = args.lookahead_distance
lookahead_indices = args.lookahead_indices
debug = args.debug
max_accel = args.max_accel
max_speed = args.max_speed
min_speed = args.min_speed
num_samples = args.num_samples

if not ( bool(lookahead_distance is not None) ^ bool(lookahead_indices is not None) ):
    raise ValueError("Either lookahead_distance or lookahead_indices (but NOT both) must be specified")

root_dir = args.db_path
with open(os.path.join(root_dir,"f1_dataset_config.yaml"),"r") as f:
    config = yaml.load(f,Loader=yaml.SafeLoader)
    use_json = config["use_json"]
motion_data_folder = os.path.join(root_dir,"udp_data","motion_packets")
image_folder = os.path.join(root_dir,"images")
session_folder = os.path.join(root_dir,"udp_data","session_packets")
session_packets = getAllSessionPackets(session_folder,use_json)
track_ids = [packet.udp_packet.m_trackId for packet in session_packets]
if(len(list(set(track_ids))) > 1):
    raise ValueError("This script only works on sessions where the whole session was done on the same track.")
track_id = track_ids[0]
output_dir = os.path.join(root_dir, args.output_dir)
if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
time.sleep(1.0)
os.makedirs(output_dir)


trackfiledir = args.trackfiledir
if trackfiledir is None:
    raise ValueError("Must either specify --trackfiledir or set environment variable F1_TRACK_DIR")

racelinepath = os.path.join(trackfiledir,trackNames[track_id] + "_racingline.json")
with open(racelinepath,"r") as f:
    racelinedict = json.load(f)
racelinein = np.row_stack((racelinedict["x"],racelinedict["y"],racelinedict["z"])).astype(np.float64)

# racelinein = np.hstack( ( racelinein , np.array([ racelinein[:,0]  ]).transpose() ) ).transpose()

dlist = racelinedict["dist"] 
# dlist.append(dlist[-1] + np.linalg.norm(finalstretch))
racelinedist = np.array(dlist, dtype=np.float64)
racelinedist = racelinedist - racelinedist[0]


print("Race distance: %f" % racelinedist[-1])
firstpoint = racelinein[:,0]
lastpoint = racelinein[:,-1]
finalstretch = lastpoint - firstpoint
finaldistance = np.linalg.norm(finalstretch)
while finaldistance==0.0:
    racelinein = racelinein[:,0:-1]
    racelinedist = racelinedist[0:-1]
    firstpoint = racelinein[:,0]
    lastpoint = racelinein[:,-1]
    finalstretch = lastpoint - firstpoint
    finaldistance = np.linalg.norm(finalstretch)
print("Distance between end and start: %s" % str(finaldistance))



plt.figure()
plt.plot(racelinein[0], racelinein[2], label="Input raceline")
plt.plot(racelinein[0,0], racelinein[2,0], 'g*', label='Raceline Start')
plt.plot(racelinein[0,-1], racelinein[2,-1], 'r*', label='Raceline End')
plt.legend()
try:
    plt.savefig(os.path.join(output_dir, "input_raceline.pdf"), format="pdf")
    plt.savefig(os.path.join(output_dir, "input_raceline.png"), format="png")
    plt.close()
except Exception as e:
    plt.close()


raceline_spline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(racelinedist, racelinein.transpose(), k=3)
raceline_tangent : scipy.interpolate.BSpline = raceline_spline.derivative(nu=1)
raceline_lateral : scipy.interpolate.BSpline = raceline_spline.derivative(nu=2)

racelineparam = np.linspace(racelinedist[0],racelinedist[-1], num=num_samples)
racelinesamp = raceline_spline(racelineparam)
racelineaug = np.row_stack( ( racelinesamp[:,0] , racelinesamp[:,1], racelinesamp[:,2], np.ones_like(racelineparam) ) ).astype(np.float64)


plt.figure()
plt.plot(racelineaug[0], racelineaug[2], label="Sampled raceline")
plt.plot(racelineaug[0,0], racelineaug[2,0], 'g*', label='Raceline Start')
plt.plot(racelineaug[0,-1], racelineaug[2,-1], 'r*', label='Raceline End')
plt.legend()
try:
    plt.savefig(os.path.join(output_dir, "sampled_raceline.pdf"), format="pdf")
    plt.savefig(os.path.join(output_dir, "sampled_raceline.png"), format="png")
    plt.close()
except Exception as e:
    plt.close()


raceline_tangents = raceline_tangent(racelineparam)
tangent_norms = np.linalg.norm(raceline_tangents, axis=1)
print(tangent_norms)
raceline_unit_tangents = raceline_tangents/tangent_norms[:,np.newaxis]
print(np.linalg.norm(raceline_unit_tangents,axis=1))

raceline_laterals = raceline_lateral(racelineparam)
raceline_lateral_norms = np.linalg.norm(raceline_laterals,axis=1)
raceline_unit_laterals = raceline_laterals/(raceline_lateral_norms[:,np.newaxis])
crossprods = np.cross(raceline_tangents, raceline_laterals)
crossprodnorms = np.linalg.norm(crossprods, axis=1)
radii = np.power(tangent_norms, 3.0)/crossprodnorms
curvatures = 1.0/radii
print("Min Radius: %f. Max Radius: %f" % (np.min(radii), np.max(radii)) )



speeds = max_speed*np.ones_like(radii, dtype=np.float64)
max_allowable_speed = np.sqrt(max_accel*radii)
print("Min Allowable Speed: %f. Max Allowable Speed: %f" % (np.min(max_allowable_speed), np.max(max_allowable_speed)) )
centripetal_accelerations = np.power(speeds,2.0)/radii
speeds[centripetal_accelerations>max_accel] = max_allowable_speed[centripetal_accelerations>max_accel]
speeds[speeds<min_speed] = min_speed
print("Min Speed: %f. Max Speed: %f" % (np.min(speeds), np.max(speeds)) )
raceline_velocities = speeds[:,np.newaxis]*raceline_unit_tangents
raceline_angular_velocity_mags = speeds/radii
raceline_angular_velocity_directions = np.cross(raceline_unit_tangents, raceline_laterals)
raceline_angular_velocity_directions = raceline_angular_velocity_directions/(np.linalg.norm(raceline_angular_velocity_directions,axis=1)[:,np.newaxis])
raceline_angular_velocities = raceline_angular_velocity_mags[:,np.newaxis]*raceline_angular_velocity_directions

#accelerations = raceline_lateral_norms*speeds


# speedspline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(racelineparam, speeds, k=3)
# dvds : scipy.interpolate.BSpline = speedspline.derivative()
# accelerations = speeds*dvds(racelineparam)
# print("Min acceleration: %f. Max acceleration: %f" % (np.min(accelerations), np.max(accelerations)))

# times = [0]
# for i in range(0, racelineaug.shape[1]-1):
#     delta_s = racelineparam[i+1] - racelineparam[i]
#     dt = delta_s/speeds[i]
#     times.append(times[-1]+dt)
# times = np.array(times)
# print(times)
# print(times.shape)


ismallest = np.argsort(speeds)
plt.figure()
plt.plot(racelineaug[0], racelineaug[2])
den = 10
plt.plot(racelineaug[0,ismallest[0:int(ismallest.shape[0]/den)]], racelineaug[2,ismallest[0:int(ismallest.shape[0]/den)]], 'g*', label='%d Points of slowest speed' % (int(ismallest.shape[0]/den),) )
plt.legend()
try:
    plt.savefig(os.path.join(output_dir, "slowest_speeds.pdf"), format="pdf")
    plt.savefig(os.path.join(output_dir, "slowest_speeds.png"), format="png")
    plt.close()
except Exception as e:
    plt.close()

plt.figure()
plt.hist(speeds, bins=25, label="Speeds")
plt.title("Speed Histogram")
try:
    plt.savefig(os.path.join(output_dir, "speed_histogram.pdf"), format="pdf")
    plt.savefig(os.path.join(output_dir, "speed_histogram.png"), format="png")
    plt.close()
except Exception as e:
    plt.close()


print(racelineaug)
spectating_flags = [bool(packet.udp_packet.m_isSpectating) for packet in session_packets]
spectating = any(spectating_flags)
car_indices = [int(packet.udp_packet.m_spectatorCarIndex) for packet in session_packets]
# print(spectating_flags)
# print(car_indices)
# print(spectating)
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

Iclip = (image_session_timestamps>(np.min(session_times) + 1.5)) * (image_session_timestamps<(np.max(session_times) - 1.5 ))
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
  plt.close()
  #text = input("Could not import matplotlib, skipping visualization. Enter anything to continue.")
#scipy.interpolate.interp1d
lmdb_dir = os.path.join(output_dir,"lmdb")
if os.path.isdir(lmdb_dir):
    shutil.rmtree(lmdb_dir)
os.makedirs(lmdb_dir)
print("Generating raceline labels")
config_dict = {"lookahead_distance": lookahead_distance}
config_dict = {"lookahead_indices": lookahead_indices}
with open(os.path.join(output_dir,'config.yaml'), 'w') as yaml_file:
    yaml.dump(config_dict, yaml_file, Dumper=yaml.SafeDumper)
#exit(0)
raceline_diffs = racelineparam[1:] - racelineparam[0:-1]
#raceline_diffs = np.linalg.norm(racelineaug[0:3,1:] - racelineaug[0:3,0:-1] , ord=2, axis=0 )
average_diff = np.mean(raceline_diffs)
if lookahead_indices is None:
    li = int(round(lookahead_distance/average_diff))
else:
    li = lookahead_indices
kdtree : KDTree = KDTree(racelineaug[0:3].transpose())
db = deepracing.backend.PoseSequenceLabelLMDBWrapper()
db.readDatabase( lmdb_dir, mapsize=int(round(166*li*len(image_tags)*1.25)), max_spare_txns=16, readonly=False )
for (idx,imagetag)  in tqdm(enumerate(image_tags)):
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
       # car_affine_pose_inv = la.inv(car_affine_pose)
        car_affine_pose_inv = np.eye(4)
        car_affine_pose_inv[0:3,0:3] = car_affine_pose[0:3,0:3].transpose()
        car_affine_pose_inv[0:3,3] = np.matmul(car_affine_pose_inv[0:3,0:3], -car_affine_pose[0:3,3]) 
        # Izpos = raceline_local[:,2]>=0.0
        # raceline_geodesic_distances = racelinedist[Izpos]
        # raceline_local = raceline_local[Izpos]
        # raceline_distances = np.linalg.norm(raceline_local,axis=1)

        (d, I1) = kdtree.query(car_position)
       # I1 = int(I1)
        I2 = I1 + li
      #  print(sample_idx)
        if I2 < racelineaug.shape[1]:
            sample_idx = np.arange(I1,I2,step=1,dtype=np.int32)
            local_distances = racelineparam[sample_idx]
            crossover = False
        else:
            crossover = True
            sample_idx1 = np.arange(I1,racelineaug.shape[1],step=1,dtype=np.int32)
            sample_idx2 = np.arange(0,I2-racelineaug.shape[1],step=1,dtype=np.int32)
            sample_idx = np.hstack((sample_idx1,sample_idx2))
            local_distances1 = racelineparam[sample_idx1]
            local_distances2 = racelineparam[sample_idx2] + (local_distances1[-1] + finaldistance)
            local_distances = np.hstack((local_distances1,local_distances2))
        local_points = np.matmul(car_affine_pose_inv,racelineaug[:,sample_idx])[0:3].transpose()
        local_velocities = np.matmul(car_affine_pose_inv[0:3,0:3],raceline_velocities[sample_idx].transpose()).transpose()
        local_speeds = speeds[sample_idx]
        local_radii = radii[sample_idx]
        local_angular_velocities = np.matmul(car_affine_pose_inv[0:3,0:3],raceline_angular_velocities[sample_idx].transpose()).transpose()

        local_times = [0.0]
        for i in range(local_distances.shape[0]-1):
            ds = local_distances[i+1] - local_distances[i]
            #ds = np.linalg.norm(local_points[i+1] - local_points[i])
            dt = ds/local_speeds[i]
            local_times.append(local_times[-1] + dt)
        local_times = np.array(local_times)
       
        if not local_times.shape[0] == li:
            raise ValueError("Local times is supposed to have %d samples, but has %d instead." % (li, local_times.shape[0]))
        if np.any((local_times[1:]-local_times[0:-1])<=0):
            raise ValueError("Local times is should be always increasing.")
        if not local_speeds.shape[0] == li:
            raise ValueError("Local speeds is supposed to have %d samples, but has %d instead." % (li, local_speeds.shape[0]))
        if not local_velocities.shape[0] == li:
            raise ValueError("Local velocities is supposed to have %d samples, but has %d instead." % (li, local_velocities.shape[0]))
        if not local_distances.shape[0] == li:
            raise ValueError("Local distances is supposed to have %d samples, but has %d instead." % (li, local_distances.shape[0]))
        if not local_points.shape[0] == li:
            raise ValueError("Local points is supposed to have %d samples, but has %d instead." % (li, local_points.shape[0]))
        if debug and (crossover or np.min(local_speeds)<40) and idx%10==0:
            print("local_points has shape: %s" % ( str(local_points.shape), ) )
            print("Total time diff %f" % ( local_times[-1] - local_times[0], ) )
            f, (imax, plotax) = plt.subplots(nrows=1 , ncols=2)
            im = cv2.cvtColor(cv2.imread(os.path.join(image_folder,label_tag.image_tag.image_file)), cv2.COLOR_BGR2RGB)
            imax.imshow(im)
            plotax.set_xlim(-40,40)
            plotax.scatter(-local_points[:,0], local_points[:,2])
            plotax.quiver(-local_points[:,0], local_points[:,2], -local_velocities[:,0], local_velocities[:,2], angles='xy', scale=None)
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
            newpose.session_time = local_times[j]
            newpose.frame = FrameId_pb2.LOCAL
            
            newvel = label_tag.subsequent_linear_velocities.add()
            newvel.vector.x = local_velocities[j,0]
            newvel.vector.y = local_velocities[j,1]
            newvel.vector.z = local_velocities[j,2]
            newvel.session_time = local_times[j]
            newvel.frame = FrameId_pb2.LOCAL

            newangvel = label_tag.subsequent_angular_velocities.add()
            newangvel.vector.x = local_angular_velocities[j,0]
            newangvel.vector.y = local_angular_velocities[j,1]
            newangvel.vector.z = local_angular_velocities[j,2]
            newangvel.session_time = local_times[j]
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





