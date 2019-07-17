import numpy as np
import numpy.linalg as la
import quaternion
import scipy
import skimage
import skimage.io
import PIL
from PIL import Image as PILImage
import TimestampedPacketMotionData_pb2
import TimestampedImageWithPose_pb2
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
import deepracing.pose_utils
import h5py
from tqdm import tqdm as tqdm
def imageDataKey(data):
    return data.timestamp
def udpPacketKey(packet):
    return packet.timestamp
clicked_row = -1
clicked_col = -1
def mouseCB(event, x, y, flags, param):
    global clicked_row, clicked_col
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_row = y
        clicked_col = x
        #print(clicked_row)
        #print(clicked_col)
parser = argparse.ArgumentParser()
parser.add_argument("motion_data_path", help="Path to motion_data packet folder",  type=str)
parser.add_argument("image_path", help="Path to image folder",  type=str)
parser.add_argument("--h5_chunks", help="Number of images to put in each H5 chunk.",  type=int, default=250)
angvelhelp = "Use the angular velocities given in the udp packets. THESE ARE ONLY PROVIDED FOR A PLAYER CAR. IF THE " +\
    " DATASET WAS TAKEN ON SPECTATOR MODE, THE ANGULAR VELOCITY VALUES WILL BE GARBAGE."
parser.add_argument("--use_given_angular_velocities", help=angvelhelp, action="store_true")
parser.add_argument("--assume_linear_timescale", help="Assumes the slope between system time and session time is 1.0", action="store_true")
parser.add_argument("--json", help="Assume dataset files are in JSON rather than binary .pb files.",  action="store_true")

args = parser.parse_args()
motion_data_folder = args.motion_data_path
image_folder = args.image_path
image_tags = deepracing.pose_utils.getAllImageFilePackets(args.image_path, args.json)
motion_packets = deepracing.pose_utils.getAllMotionPackets(args.motion_data_path, args.json)
motion_packets = sorted(motion_packets, key=udpPacketKey)
session_times = np.array([packet.udp_packet.m_header.m_sessionTime for packet in motion_packets])
system_times = np.array([packet.timestamp/1000.0 for packet in motion_packets])
print(system_times)
print(session_times)
maxudptime = system_times[-1]
image_tags = [tag for tag in image_tags if tag.timestamp/1000.0<maxudptime]
image_tags = sorted(image_tags, key = imageDataKey)
image_timestamps = np.array([data.timestamp/1000.0 for data in image_tags])
image_dt = np.diff(image_timestamps)
mean_dt = np.mean(image_dt)
mean_freq = 1/mean_dt
print("Average image dt: %f. 1/dt:= %f" % (mean_dt, mean_freq))


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

poses = [deepracing.pose_utils.extractPose(packet.udp_packet) for packet in motion_packets]
velocities = np.array([deepracing.pose_utils.extractVelocity(packet.udp_packet) for packet in motion_packets])
positions = np.array([pose[0] for pose in poses])
quaternions = np.array([pose[1] for pose in poses])
if args.use_given_angular_velocities:
    angular_velocities = np.array([deepracing.pose_utils.extractAngularVelocity(packet.udp_packet) for packet in motion_packets])
else:
    angular_velocities = quaternion.angular_velocity(quaternions, session_times)

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
angular_velocity_interpolant = scipy.interpolate.interp1d(session_times, angular_velocities, axis=0, kind='cubic')
interpolated_positions = position_interpolant(image_session_timestamps)
interpolated_velocities = velocity_interpolant(image_session_timestamps)
interpolated_quaternions = quaternion.squad(quaternions, session_times, image_session_timestamps)
interpolated_angular_velocities2 = angular_velocity_interpolant(image_session_timestamps)
interpolated_angular_velocities = quaternion.angular_velocity(interpolated_quaternions, image_session_timestamps)
angvel_diff = interpolated_angular_velocities2 - interpolated_angular_velocities
diff_norms = la.norm(angvel_diff,axis=1)
print("Mean diff between angular velocity techniques: %f" %(np.mean(diff_norms)))
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
except Exception as e:
    print(str(e))
    #input("Enter anything to continue\n")
#scipy.interpolate.interp1d
label_folder = "pose_labels"
if(not os.path.isdir(os.path.join(image_folder,label_folder))):
    os.makedirs(os.path.join(image_folder,label_folder))
dsfile = os.path.join(image_folder,'h5dataset.hdf5')
#prev_img = cv2.imread(os.path.join(image_folder,image_tags[0].image_file))
prev_img = skimage.util.img_as_ubyte(skimage.io.imread(os.path.join(image_folder,image_tags[0].image_file)))

input("Enter anything to continue\n")
try:
    skimage.io.imshow(prev_img)
    skimage.io.show()
    #cv2.namedWindow("first_image",cv2.WINDOW_AUTOSIZE)
    #cv2.imshow("first_image",cv2.cvtColor(prev_img, cv2.COLOR_RGB2BGR))
    #cv2.waitKey(0)
    #cv2.destroyWindow("first_image")
except Exception as e:
    print(str(e))

if(os.path.isfile(dsfile)):
    os.remove(dsfile)
hf5file = h5py.File(dsfile, 'w')
dsetlen = len(image_tags)
image_chunks = None
if args.h5_chunks>0:
    image_chunks = (args.h5_chunks,prev_img.shape[0],prev_img.shape[1],prev_img.shape[2])
label_chunks = None
label_rotation_chunks = None
image_dset = hf5file.create_dataset("images", chunks=image_chunks, shape=(dsetlen,prev_img.shape[0],prev_img.shape[1],prev_img.shape[2]), dtype='uint8')
position_dset = hf5file.create_dataset("position", chunks=label_chunks, shape=(dsetlen,3), dtype='float64')
rotation_dset = hf5file.create_dataset("rotation", chunks=label_rotation_chunks, shape=(dsetlen,4), dtype='float64')
linear_velocity_dset = hf5file.create_dataset("linear_velocity", chunks=label_chunks,shape=(dsetlen,3), dtype='float64')
angular_velocity_dset = hf5file.create_dataset("angular_velocity", chunks=label_chunks, shape=(dsetlen,3), dtype='float64')
session_time_dset = hf5file.create_dataset("session_time", chunks=label_chunks, shape=(dsetlen,), dtype='float64')

for idx in tqdm(range(dsetlen)):
    label_tag = TimestampedImageWithPose_pb2.TimestampedImageWithPose()
    label_tag.timestamped_image.CopyFrom(image_tags[idx])
    label_tag.pose.frame = FrameId_pb2.GLOBAL
    label_tag.linear_velocity.frame = FrameId_pb2.GLOBAL
    label_tag.angular_velocity.frame = FrameId_pb2.GLOBAL

    

    t_interp = image_session_timestamps[idx]
    label_tag.pose.session_time = t_interp
    label_tag.linear_velocity.session_time = t_interp
    label_tag.angular_velocity.session_time = t_interp

    carposition_global = interpolated_positions[idx]
    #carposition_global = interpolateVectors(pos1,session_times[i-1],pos2,session_times[i], t_interp)
    carvelocity_global = interpolated_velocities[idx]
    
    #carvelocity_global = interpolateVectors(vel1,session_times[i-1],vel2,session_times[i], t_interp)
    carquat_global = interpolated_quaternions[idx]
    carquat_global = carquat_global/carquat_global.norm()
    #carquat_global = quaternion.slerp(quat1,quat2,session_times[i-1],session_times[i], t_interp)
    carangvelocity_global = interpolated_angular_velocities[idx]
    #carangvelocity_global = interpolateVectors(angvel1,session_times[i-1],angvel2,session_times[i], t_interp)
    #position_dset[idx]=carposition_global
    #rotation_dset[idx]=quaternion.as_float_array(carquat_global)
    #linear_velocity_dset[idx]=carvelocity_global
    #angular_velocity_dset[idx]=carangvelocity_global
    #session_time_dset[idx]=t_interp
    dset_index = idx
    curr_img = skimage.util.img_as_ubyte(skimage.io.imread(os.path.join(image_folder,image_tags[idx].image_file)))
    #curr_img = cv2.imread(os.path.join(image_folder,image_tags[idx].image_file))
    image_dset[dset_index] = curr_img
    rotation_dset[dset_index] = quaternion.as_float_array(carquat_global)
    position_dset[dset_index] = carposition_global
    linear_velocity_dset[dset_index] = carvelocity_global
    angular_velocity_dset[dset_index] =  carangvelocity_global
    session_time_dset[dset_index] =  t_interp
    hf5file.flush()
        
    label_tag.pose.translation.x = carposition_global[0]
    label_tag.pose.translation.y = carposition_global[1]
    label_tag.pose.translation.z = carposition_global[2]
    label_tag.pose.rotation.x = carquat_global.x
    label_tag.pose.rotation.y = carquat_global.y
    label_tag.pose.rotation.z = carquat_global.z
    label_tag.pose.rotation.w = carquat_global.w

    label_tag.linear_velocity.vector.x = carvelocity_global[0]
    label_tag.linear_velocity.vector.y = carvelocity_global[1]
    label_tag.linear_velocity.vector.z = carvelocity_global[2]
    
    label_tag.angular_velocity.vector.x = carangvelocity_global[0]
    label_tag.angular_velocity.vector.y = carangvelocity_global[1]
    label_tag.angular_velocity.vector.z = carangvelocity_global[2]

    label_tag_JSON = google.protobuf.json_format.MessageToJson(label_tag, including_default_value_fields=True)
    image_file_base = os.path.splitext(os.path.split(label_tag.timestamped_image.image_file)[1])[0]
    label_tag_file_path = os.path.join(image_folder,label_folder,image_file_base + "_pose_label.json")
   # print(label_tag_JSON)
    f = open(label_tag_file_path,'w')
    f.write(label_tag_JSON)
    f.close()
    label_tag_file_path = os.path.join(image_folder,label_folder,image_file_base + "_pose_label.pb")
    f = open(label_tag_file_path,'wb')
    f.write(label_tag.SerializeToString())
    f.close()
    prev_img = curr_img
    #print(carquatinverse)
   # print(carquat)
hf5file.close()




