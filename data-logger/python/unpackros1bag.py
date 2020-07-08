import rosbag
import yaml
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, LaserEcho, Image, CompressedImage
import argparse
import TimestampedImage_pb2, TimestampedPacketMotionData_pb2, PoseSequenceLabel_pb2, Image_pb2, FrameId_pb2
import Pose3d_pb2, Vector3dStamped_pb2

from tqdm import tqdm as tqdm
import rospy
from rospy import Time
import numpy as np, scipy
import scipy, scipy.integrate, scipy.interpolate
from scipy.spatial.transform import Rotation as Rot, RotationSpline as RotSpline
import matplotlib
from matplotlib import pyplot as plt
import cv_bridge
import cv2
import shutil
import os
import time
import deepracing, deepracing.backend, deepracing.protobuf_utils as proto_utils
import google.protobuf.json_format
import bisect
# /car_1/camera/image_raw      8263 msgs    : sensor_msgs/Image
# /car_1/multiplexer/command   7495 msgs    : ackermann_msgs/AckermannDrive
# /car_1/odom_filtered         6950 msgs    : nav_msgs/Odometry
# /car_1/scan                  9870 msgs    : sensor_msgs/LaserScan
parser = argparse.ArgumentParser("Unpack a bag file into a dataset")
parser.add_argument('bagfile', type=str,  help='The bagfile to unpack')
parser.add_argument('--config', type=str, required=True , help="Config file specifying the rostopics to unpack")
parser.add_argument('--lookahead_indices', type=int, default=60, help="Number of indices to look forward")
parser.add_argument('--debug', action="store_true", help="Display some debug plots")
args = parser.parse_args()
argdict = vars(args)
lookahead_indices = argdict["lookahead_indices"]
configfile = argdict["config"]
with open(configfile,'r') as f:
    topicdict : dict =yaml.load(f, Loader=yaml.SafeLoader)
bagpath = argdict["bagfile"]
debug = argdict["debug"]
bag = rosbag.Bag(bagpath)
msg_types, typedict = bag.get_type_and_topic_info()
print(typedict)
imagetypestr = typedict[topicdict["images"]].msg_type
if  imagetypestr == "sensor_msgs/Image":
    compressed = False
elif imagetypestr == "sensor_msgs/CompressedImage":
    compressed = True
else:
    raise ValueError("Invalid image type %s on topic %s" % (imagetypestr , topicdict["images"] ) )

print()

topics = list(topicdict.values())

odomtimes = []
imagetimes = []

odomsunsorted = []
imagemsgsunsorted = []
for topic, msg, t in tqdm(iterable=bag.read_messages(topics=topics), desc="Loading messages from bag file"):
    if topic==topicdict["images"]:
        stamp = msg.header.stamp
        imagemsgsunsorted.append(msg)
        imagetimes.append(Time(secs = stamp.secs, nsecs=stamp.nsecs).to_sec())
    elif topic==topicdict["odom"]:
        stamp = msg.header.stamp
        odomsunsorted.append(msg)
        odomtimes.append(Time(secs = stamp.secs, nsecs=stamp.nsecs).to_sec())
bag.close()
imagetimes = np.array(imagetimes).astype(np.float64)
odomtimes = np.array(odomtimes).astype(np.float64)
Iodomtimes = np.argsort(odomtimes)
Iimagetimes = np.argsort(imagetimes)
print(Iodomtimes)
print(Iimagetimes)

imagemsgs = [imagemsgsunsorted[Iimagetimes[i]] for i in range(len(Iimagetimes))]
imagetimes = imagetimes[Iimagetimes]

odoms = [odomsunsorted[Iodomtimes[i]] for i in range(len(Iodomtimes))]
odomtimes = odomtimes[Iodomtimes]

Iclip = (imagetimes>odomtimes[0]) * (imagetimes<odomtimes[-1])
imagemsgs = [imagemsgs[i] for i in range(len(Iclip)) if Iclip[i]]
imagetimes = imagetimes[Iclip]

t0 = odomtimes[0]
odomtimes = odomtimes - t0
imagetimes = imagetimes - t0

odomtimes01 = odomtimes/odomtimes[-1]
imagetimes01 = imagetimes/odomtimes[-1]

poses = [odom.pose.pose for odom in odoms]
print(poses[0])
positionsmsg = [pose.position for pose in poses]
rotationsmsg = [pose.orientation for pose in poses]

positions = np.array([[p.x,p.y,0.0] for p in positionsmsg]).astype(np.float64)
quaternions = np.array([[q.x,q.y,q.z,q.w] for q in rotationsmsg]).astype(np.float64)
rotations = Rot.from_quat(quaternions)


positionspline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(odomtimes01, positions)
velocityspline : scipy.interpolate.BSpline = positionspline.derivative()
rotationspline = RotSpline(odomtimes01, rotations)

imagepositions = positionspline(imagetimes01)
imagevelocities = velocityspline(imagetimes01)/odomtimes[-1]

imagerotations = rotationspline(imagetimes01)
imagequaternions = imagerotations.as_quat()
imagerotationmatrices = imagerotations.as_matrix()
imageangularvelocities = rotationspline(imagetimes01,order=1)/odomtimes[-1]


try:
    plt.plot(positions[:,0], positions[:,1], c="b", label="Recorded points")
    plt.show()
except:
    pass



print(positions[0])
print(quaternions[0])
print(imagepositions.shape)
print(imagequaternions.shape)
print(imageangularvelocities)
bridge : cv_bridge.CvBridge = cv_bridge.CvBridge()
images = []
for i in tqdm(iterable=range(len(imagemsgs)), desc="Converting images to numpy arrays"):
    imgmsg = imagemsgs[i]
    if compressed:
        images.append(bridge.compressed_imgmsg_to_cv2(imagemsgs[i],desired_encoding="bgr8"))
    else:
        images.append(bridge.imgmsg_to_cv2(imagemsgs[i],desired_encoding="bgr8"))

#images = np.array(images)
bagdir = os.path.dirname(bagpath)
rootdir = os.path.join(bagdir, os.path.splitext(os.path.basename(bagpath))[0])

imagedir = os.path.join(rootdir,"images")
os.makedirs(imagedir,exist_ok=True)

labeldir = os.path.join(rootdir,"pose_sequence_labels")
os.makedirs(labeldir,exist_ok=True)

lmdb_dir = os.path.join(labeldir,"lmdb")
if os.path.isdir(lmdb_dir):
    shutil.rmtree(lmdb_dir)
time.sleep(0.5)
os.makedirs(lmdb_dir)
#cv2.namedWindow("image", flags=cv2.WINDOW_AUTOSIZE)
db = deepracing.backend.PoseSequenceLabelLMDBWrapper()
db.readDatabase( lmdb_dir, mapsize=int(round(9996*len(images)*1.25)), max_spare_txns=16, readonly=False, lock=True )
keys = ["image_%d" % (i+1,) for i in range(len(images))]
lstsquare_spline_degree = 7
localsplinedelta = 10
for i in tqdm(iterable=range(len(images)), desc="Writing images to file"):
    key = keys[i]
    keyfile = key + ".jpg" 
    filename = os.path.join(imagedir, keyfile)
    cv2.imwrite(filename,images[i])
    imax = i+lookahead_indices
    if imax>=len(images):
        continue
    label_tag : PoseSequenceLabel_pb2.PoseSequenceLabel = PoseSequenceLabel_pb2.PoseSequenceLabel()
    label_tag.image_tag.timestamp = imagetimes[i]
    label_tag.image_tag.image_file = keyfile
    label_tag.car_pose.frame = FrameId_pb2.GLOBAL
    label_tag.car_velocity.frame = FrameId_pb2.GLOBAL
    label_tag.car_angular_velocity.frame = FrameId_pb2.GLOBAL
    iclosest = bisect.bisect_left(odomtimes, label_tag.image_tag.timestamp)
    #bisect.bisect_right
    odomidxmin = iclosest - localsplinedelta
    odomidxmax = iclosest + lookahead_indices + localsplinedelta
    if odomidxmin<0 or odomidxmax>=len(odomtimes):
        continue
    positionslocal = np.array(positions[odomidxmin:odomidxmax])
    rotationslocal = rotations[odomidxmin:odomidxmax]
    odomtimeslocal = np.array(odomtimes[odomidxmin:odomidxmax])
   # odomtimeslocal = (odomtimeslocal - odomtimeslocal[0])/(odomtimeslocal[-1]-odomtimeslocal[0])
    numsamples = odomtimeslocal.shape[0]
    t = [ odomtimeslocal[int(numsamples/4)], odomtimeslocal[int(numsamples/2)], odomtimeslocal[int(3*numsamples/4)] ]
    t = np.r_[(odomtimeslocal[0],)*(lstsquare_spline_degree+1),  t,  (odomtimeslocal[-1],)*(lstsquare_spline_degree+1)]
    # print(odomtimeslocal)
    # print(t)
    positionsplinelocal : scipy.interpolate.BSpline = scipy.interpolate.make_lsq_spline(odomtimeslocal, positionslocal[:,[0,1]], t, k = lstsquare_spline_degree)
    velocitysplinelocal : scipy.interpolate.BSpline = positionsplinelocal.derivative()
    rotationsplinelocal : RotSpline = RotSpline(odomtimeslocal, rotationslocal)
    tsamp = np.linspace(label_tag.image_tag.timestamp, odomtimeslocal[-localsplinedelta], num=lookahead_indices)
    position_samples = positionsplinelocal(tsamp)
    velocity_samples = velocitysplinelocal(tsamp)
    rotation_samples = rotationsplinelocal(tsamp)
    angvel_samples = rotationsplinelocal(tsamp, order=1)
    rotation_samples_matrices = rotation_samples.as_matrix()

   # print("velocity_samples.shape: " + str(velocity_samples.shape))
    forward = np.hstack((velocity_samples[0],np.array([0]))).copy()
    forward = forward/np.linalg.norm(forward)
    up = np.array((0.0,0.0,1.0))
    left = np.cross(up,forward)
    left = left/np.linalg.norm(left)
   # print("angle between forward and left: " + str(180/np.pi*np.arccos(np.dot(forward,left))))

    carpose = np.eye(4).astype(np.float64)
    # carpose[0:3,0:3] = imagerotationmatrices[i]
    # carpose[0:3,3] = imagepositions[i]
    carpose[0:2,3] = position_samples[0]
    #carpose[0:3,0:3] = rotation_samples_matrices[0]
    carpose[0:3,0] = forward
    carpose[0:3,1] = left
    carpose[0:3,2] = up
    carquat = Rot.from_matrix(carpose[0:3,0:3]).as_quat()
    carquat[np.abs(carquat)<1E-12]=0.0
    carposeinv = np.linalg.inv(carpose)
    carposeinvrotmat = carposeinv[0:3,0:3]

    labelposesglobal = np.array([np.eye(4).astype(np.float64) for i in range(lookahead_indices)])
    # labelposesglobal[:,0:3,0:3] = odom[i:imax]
    #labelposesglobal[:,0:3,0:3] = rotation_samples_matrices
    for j in range(lookahead_indices):
        forward = np.hstack((velocity_samples[j],np.array([0.0]))).copy()
        forward = forward/np.linalg.norm(forward)
        up = np.array((0.0,0.0,1.0))
        left = np.cross(up,forward)
        left = left/np.linalg.norm(left)
        labelposesglobal[j,0:3,0] = forward
        labelposesglobal[j,0:3,1] = left
        labelposesglobal[j,0:3,2] = up
    
    #labelposesglobal[:,0:3,3] = positionslocal[localsplinedelta:-localsplinedelta]
    labelposesglobal[:,0:2,3] = position_samples
    

    # labelvelsglobal = imagevelocities[i:imax].transpose()
    labelvelsglobal = velocity_samples.transpose()

    #labelangvelsglobal = imageangularvelocities[i:imax].transpose()
    labelangvelsglobal = angvel_samples.transpose()
    # print("labelvelsglobal.shape: " + str(labelvelsglobal.shape))
    # print("carposeinv.shape: " + str(carposeinv.shape))
    # print(labelangvelsglobal.shape)
    # print(carposeinvrotmat.shape)
    labelposes = np.matmul(carposeinv,labelposesglobal)
    labelpositions = labelposes[:,0:3,3]
    labelrotmats = labelposes[:,0:3,0:3]
    labelrots = Rot.from_matrix(labelrotmats)
    labelquats = labelrots.as_quat()
    labelquats[np.abs(labelquats)<1E-12]=0.0

    #labelvels = np.matmul(carposeinvrotmat,labelvelsglobal)
    labelvelsglobalaug = np.vstack((labelvelsglobal[0], labelvelsglobal[1] ,np.zeros_like(labelvelsglobal[0])))
    #print("labelvelsglobalaug.shape: " + str(labelvelsglobalaug.shape))
    labelvels = np.matmul(carposeinvrotmat,labelvelsglobalaug)
    labelangvels = np.matmul(carposeinvrotmat,labelangvelsglobal)
    label_tag = PoseSequenceLabel_pb2.PoseSequenceLabel()
    t_interp = imagetimes[i]
    label_tag.car_pose.session_time = t_interp
    label_tag.car_velocity.session_time = t_interp
    label_tag.car_angular_velocity.session_time = t_interp

    label_tag.car_pose.translation.CopyFrom(proto_utils.vectorFromNumpy(carpose[:,3]))
    label_tag.car_pose.rotation.CopyFrom(proto_utils.quaternionFromNumpy(carquat))
    label_tag.car_velocity.vector.CopyFrom(proto_utils.vectorFromNumpy(labelvelsglobalaug[:,0]))
    label_tag.car_angular_velocity.vector.CopyFrom(proto_utils.vectorFromNumpy(labelangvelsglobal[:,0]))
    if debug:
        plt.scatter(positionslocal[:,0], positionslocal[:,1], label='measured data', facecolors='none', edgecolors='b', s=8)
        plt.plot(labelposesglobal[:,0,3], labelposesglobal[:,1,3], label='samples from spline fit', c='r')
        plt.plot(carpose[0,3], carpose[1,3], 'g*', label='current car position')
        
        plt.arrow(carpose[0,3], carpose[1,3], 0.25*carpose[0,0], 0.25*carpose[1,0])
       # plt.arrow(carpose[0,3], carpose[1,3], 0.25*carpose[0,1], 0.25*carpose[1,1])
        #WX = np.
        #plt.quiver(X, Y, dX, dY, color='r')
        #plt.quiver([], [], [], [])
        plt.show()

    for j in range(lookahead_indices):
        pose_forward_pb = Pose3d_pb2.Pose3d()
        newpose = label_tag.subsequent_poses.add()
        newvel = label_tag.subsequent_linear_velocities.add()
        newangvel = label_tag.subsequent_angular_velocities.add()
        newpose.frame = FrameId_pb2.LOCAL
        newvel.frame = FrameId_pb2.LOCAL
        newangvel.frame = FrameId_pb2.LOCAL

        newpose.translation.CopyFrom(proto_utils.vectorFromNumpy(labelpositions[j]))
        newpose.rotation.CopyFrom(proto_utils.quaternionFromNumpy(labelquats[j]))

        newvel.vector.CopyFrom(proto_utils.vectorFromNumpy(labelvels[:,j]))

        newangvel.vector.CopyFrom(proto_utils.vectorFromNumpy(labelangvels[:,j]))

        #labeltime = imagetimes[i+j]
        #labeltime = odomtimes[iclosest+j]
        labeltime = tsamp[j]
       # tsamp
        newpose.session_time = labeltime
        newvel.session_time = labeltime
        newangvel.session_time = labeltime
    label_tag_file_path = os.path.join(labeldir, key + "_sequence_label.json")
    with open(label_tag_file_path,'w') as f:
        f.write(google.protobuf.json_format.MessageToJson(label_tag, including_default_value_fields=True))
    label_tag_file_path_binary = os.path.join(labeldir, key + "_sequence_label.pb")
    with open(label_tag_file_path_binary,'wb') as f:
        f.write( label_tag.SerializeToString() )
    
    image_tag_file_path = os.path.join(imagedir, key + ".json")
    with open(image_tag_file_path,'w') as f:
        f.write(google.protobuf.json_format.MessageToJson(label_tag.image_tag, including_default_value_fields=True))
    image_tag_file_path_binary = os.path.join(imagedir, key + ".pb")
    with open(image_tag_file_path_binary,'wb') as f:
        f.write( label_tag.image_tag.SerializeToString() )
    
    db.writePoseSequenceLabel(key, label_tag)

with open(os.path.join(rootdir,"goodkeys.txt"),"w") as f:
    for i in tqdm(iterable=range(len(keys) - lookahead_indices), desc="Checking for invalid keys"):
        key = keys[i]
        try:
            labelout = db.getPoseSequenceLabel(key)
            if labelout is None:
                raise ValueError("Resulting label from key %s is None" %(key,))
        except KeyboardInterrupt as ekey:
            exit(0)
        except Exception as e:
            print("Skipping invalid key %s because of exception %s" % (key, str(e)))
            continue
        f.write(key + "\n")