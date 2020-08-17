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
import PIL
from PIL import Image, ImageFilter, ImageDraw
# /car_1/camera/image_raw      8263 msgs    : sensor_msgs/Image
# /car_1/multiplexer/command   7495 msgs    : ackermann_msgs/AckermannDrive
# /car_1/odom_filtered         6950 msgs    : nav_msgs/Odometry
# /car_1/scan                  9870 msgs    : sensor_msgs/LaserScan
parser = argparse.ArgumentParser("Unpack a bag file into a dataset")
parser.add_argument('bagfile', type=str,  help='The bagfile to unpack')
parser.add_argument('--config', type=str, required=False, default=None , help="Config file specifying the rostopics to unpack, defaults to a file named \"topicconfig.yaml\" in the same directory as the bagfile")
parser.add_argument('--lookahead_indices', type=int, default=30, help="Number of indices to look forward")
parser.add_argument('--sample_indices', type=int, default=60, help="Number of points to sample for the label")
parser.add_argument('--debug', action="store_true", help="Display some debug plots")
parser.add_argument('--mintime', type=float, default=5.0, help="Ignore this many seconds of data at the beginning of the bag file")
args = parser.parse_args()
argdict = vars(args)
lookahead_indices = argdict["lookahead_indices"]
sample_indices = argdict["sample_indices"]
configfile = argdict["config"]
bagpath = argdict["bagfile"]
mintime = argdict["mintime"]
debug = argdict["debug"]
bagdir = os.path.dirname(bagpath)
if configfile is None:
    configfile = os.path.join(bagdir,"topicconfig.yaml")
with open(configfile,'r') as f:
    topicdict : dict =yaml.load(f, Loader=yaml.SafeLoader)
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
imagemsgsunsorted = []
positionsmsg_unsorted = []
rotationsmsg_unsorted = []
for topic, msg, t in tqdm(iterable=bag.read_messages(topics=topics), desc="Loading messages from bag file", total=bag.get_message_count()):
    if topic==topicdict["images"]:
        stamp = msg.header.stamp
        imagemsgsunsorted.append(msg)
        imagetimes.append(Time(secs = stamp.secs, nsecs=stamp.nsecs).to_sec())
    elif topic==topicdict.get("odom", None):
        stamp = msg.header.stamp
        positionsmsg_unsorted.append(msg.pose.pose.position)
        rotationsmsg_unsorted.append(msg.pose.pose.orientation)
        odomtimes.append(Time(secs = stamp.secs, nsecs=stamp.nsecs).to_sec())
    elif topic==topicdict.get("tf", None):
        transform = None
       # print(msg)
        transforms = msg.transforms
        for i in range(len(transforms)):
            if transforms[i].header.frame_id=="/map" and transforms[i].child_frame_id=="/car_1_base_link":
                transform = transforms[i]
                break
        if transform is not None:
            stamp = transform.header.stamp
            odomtimes.append(Time(secs = stamp.secs, nsecs = stamp.nsecs).to_sec())
            positionsmsg_unsorted.append(transform.transform.translation)
            rotationsmsg_unsorted.append(transform.transform.rotation)
bag.close()
imagetimes = np.array(imagetimes).astype(np.float64)
odomtimes = np.array(odomtimes).astype(np.float64)
Iodomtimes = np.argsort(odomtimes)
Iimagetimes = np.argsort(imagetimes)
print(Iodomtimes)
print(Iimagetimes)

imagemsgs = [imagemsgsunsorted[Iimagetimes[i]] for i in range(len(Iimagetimes))]
imagetimes = imagetimes[Iimagetimes]

positionsmsg = [positionsmsg_unsorted[Iodomtimes[i]] for i in range(len(Iodomtimes))]
rotationsmsg = [rotationsmsg_unsorted[Iodomtimes[i]] for i in range(len(Iodomtimes))]
odomtimes = odomtimes[Iodomtimes]

Iclip = (imagetimes>odomtimes[0]) * (imagetimes<odomtimes[-1])
imagemsgs = [imagemsgs[i] for i in range(len(Iclip)) if Iclip[i]]
imagetimes = imagetimes[Iclip]

t0 = odomtimes[0]
odomtimes = odomtimes - t0
imagetimes = imagetimes - t0

odomtimes01 = odomtimes/odomtimes[-1]
imagetimes01 = imagetimes/odomtimes[-1]

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


rootdir = os.path.join(bagdir, os.path.splitext(os.path.basename(bagpath))[0])

print(positions[0])
print(quaternions[0])
print(imagepositions.shape)
print(imagequaternions.shape)
print(imageangularvelocities)

#images = np.array(images)

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
keys = ["image_%d" % (i+1,) for i in range(len(imagemsgs))]
db = deepracing.backend.PoseSequenceLabelLMDBWrapper()
db.readDatabase( lmdb_dir, mapsize=int(round(9996*len(imagemsgs)*1.25)), max_spare_txns=16, readonly=False, lock=True )
lstsquare_spline_degree = 5
localsplinedelta = 10
bridge : cv_bridge.CvBridge = cv_bridge.CvBridge()
imgfreq = typedict[topicdict["images"]].frequency
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
videowriter = None 
for (i,key) in tqdm(iterable=enumerate(keys), desc="Writing images to file", total=len(keys)):
    label_tag : PoseSequenceLabel_pb2.PoseSequenceLabel = PoseSequenceLabel_pb2.PoseSequenceLabel()
    label_tag.image_tag.timestamp = imagetimes[i]
    label_tag.image_tag.image_file = key + ".jpg" 
    if compressed:
        imcv = bridge.compressed_imgmsg_to_cv2(imagemsgs[i],desired_encoding="bgr8")
    else:
        imcv = bridge.imgmsg_to_cv2(imagemsgs[i],desired_encoding="bgr8")
    if videowriter is None:
        videowriter = cv2.VideoWriter(os.path.join(rootdir, "video.avi"), fourcc, int(round(imgfreq)), (imcv.shape[1], imcv.shape[0]), True)
    imgpil = Image.fromarray(imcv.copy())
    imdraw = ImageDraw.ImageDraw(imgpil)
    imdraw.text((imcv.shape[1]/2, imcv.shape[0]/2),"image_%d"%(i,), fill=(0,0,0))
    videowriter.write(np.asarray(imgpil))
    filename = os.path.join(imagedir, label_tag.image_tag.image_file)
    cv2.imwrite(filename,imcv)

    image_tag_file_path = os.path.join(imagedir, key + ".json")
    with open(image_tag_file_path,'w') as f:
        f.write(google.protobuf.json_format.MessageToJson(label_tag.image_tag, including_default_value_fields=True))
    image_tag_file_path_binary = os.path.join(imagedir, key + ".pb")
    with open(image_tag_file_path_binary,'wb') as f:
        f.write( label_tag.image_tag.SerializeToString() )

    imax = i+lookahead_indices
    if imax>=len(imagemsgs) or label_tag.image_tag.timestamp<=mintime or label_tag.image_tag.timestamp>=odomtimes[-1]-mintime:
        continue
    label_tag.car_pose.frame = FrameId_pb2.GLOBAL
    label_tag.car_velocity.frame = FrameId_pb2.GLOBAL
    label_tag.car_angular_velocity.frame = FrameId_pb2.GLOBAL
    iclosest = bisect.bisect_left(odomtimes, label_tag.image_tag.timestamp)
    #bisect.bisect_right
    odomidxmin = iclosest - localsplinedelta
    odomidxmax = iclosest + lookahead_indices + localsplinedelta
    if odomidxmin<0 or odomidxmax>=len(odomtimes):
        continue
    positionsglobal = np.array(positions[odomidxmin:odomidxmax])
    rotationsglobal = rotations[odomidxmin:odomidxmax]
    odomtimesglobal = np.array(odomtimes[odomidxmin:odomidxmax])
   # odomtimeslocal = (odomtimeslocal - odomtimeslocal[0])/(odomtimeslocal[-1]-odomtimeslocal[0])
    numsamples = odomtimesglobal.shape[0]
    t = [ odomtimesglobal[int(numsamples/4)], odomtimesglobal[int(numsamples/2)], odomtimesglobal[int(3*numsamples/4)] ]
    t = np.r_[(odomtimesglobal[0],)*(lstsquare_spline_degree+1),  t,  (odomtimesglobal[-1],)*(lstsquare_spline_degree+1)]
    # print(odomtimeslocal)
    # print(t)
    positionsplineglobal : scipy.interpolate.BSpline = scipy.interpolate.make_lsq_spline(odomtimesglobal, positionsglobal[:,[0,1]], t, k = lstsquare_spline_degree)
    velocitysplineglobal : scipy.interpolate.BSpline = positionsplineglobal.derivative()
    rotationsplineglobal : RotSpline = RotSpline(odomtimesglobal, rotationsglobal)
    tsamp = np.linspace(label_tag.image_tag.timestamp, odomtimesglobal[-localsplinedelta], num=sample_indices)
    position_samples = positionsplineglobal(tsamp)
    velocity_samples = velocitysplineglobal(tsamp)
    rotation_samples = rotationsplineglobal(tsamp)
    angvel_samples = rotationsplineglobal(tsamp, order=1)
    rotation_samples_matrices = rotation_samples.as_matrix()

   # print("velocity_samples.shape: " + str(velocity_samples.shape))
    v0list = list(velocity_samples[0])
    v0list.append(0)
    forward = np.array(v0list)
    forward = forward/np.linalg.norm(forward)
    # if forward[0]<=0:
    #     forward = -forward
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
    #carquat[np.abs(carquat)<1E-12]=0.0
    carposeinv = np.linalg.inv(carpose)
    carposeinvrotmat = carposeinv[0:3,0:3]

    labelposesglobal = np.array([np.eye(4).astype(np.float64) for i in range(sample_indices)])
    positionsglobalaug = np.zeros((4,positionsglobal.shape[0])).astype(np.float64)
    positionsglobalaug[3,:] = 1.0
    positionsglobalaug[0,:] = positionsglobal[:,0]
    positionsglobalaug[1,:] = positionsglobal[:,1]
    # labelposesglobal[:,0:3,0:3] = odom[i:imax]
    #labelposesglobal[:,0:3,0:3] = rotation_samples_matrices
    for j in range(labelposesglobal.shape[0]):
        vlist = list(velocity_samples[j])
        vlist.append(0)
        forward = np.array(vlist)
        forward = forward/np.linalg.norm(forward)
        left = np.cross(up,forward)
        left = left/np.linalg.norm(left)
        labelposesglobal[j,0:3,0] = forward
        labelposesglobal[j,0:3,1] = left
        labelposesglobal[j,0:3,2] = up
    
    #pfpositionsglobal[:,0:3,3] = positionslocal[localsplinedelta:-localsplinedelta]
    labelposesglobal[:,0:2,3] = position_samples
    

    # labelvelsglobal = imagevelocities[i:imax].transpose()
    labelvelsglobal = velocity_samples.transpose()

    #labelangvelsglobal = imageangularvelocities[i:imax].transpose()
    labelangvelsglobal = angvel_samples.transpose()
    # print("labelvelsglobal.shape: " + str(labelvelsglobal.shape))
    # print("carposeinv.shape: " + str(carposeinv.shape))
    # print(labelangvelsglobal.shape)
    # print(carposeinvrotmat.shape)
    positionslocal = np.matmul(carposeinv,positionsglobalaug)[0:3,:].transpose()
    labelposes = np.matmul(carposeinv,labelposesglobal)
    labelpositions = labelposes[:,0:3,3]
    labelrotmats = labelposes[:,0:3,0:3]
    labelrots = Rot.from_matrix(labelrotmats)
    labelquats = labelrots.as_quat()
    labelquats[np.abs(labelquats)<1E-12]=0.0

    #labelvels = np.matmul(carposeinvrotmat,labelvelsglobal)
    labelvelsglobal = np.zeros((sample_indices,3))
    labelvelsglobal[:,[0,1]] = velocity_samples
    #print("labelvelsglobalaug.shape: " + str(labelvelsglobalaug.shape))
    labelvels = np.matmul(carposeinvrotmat,labelvelsglobal.transpose()).transpose()
    labelangvels = np.matmul(carposeinvrotmat,labelangvelsglobal).transpose()
    label_tag.car_pose.session_time = label_tag.image_tag.timestamp
    label_tag.car_velocity.session_time = label_tag.image_tag.timestamp
    label_tag.car_angular_velocity.session_time = label_tag.image_tag.timestamp

    label_tag.car_pose.translation.CopyFrom(proto_utils.vectorFromNumpy(carpose[:,3]))
    label_tag.car_pose.rotation.CopyFrom(proto_utils.quaternionFromNumpy(carquat))
    label_tag.car_velocity.vector.CopyFrom(proto_utils.vectorFromNumpy(labelvelsglobal[0]))
    label_tag.car_angular_velocity.vector.CopyFrom(proto_utils.vectorFromNumpy(labelangvelsglobal[:,0]))
    transformnorms = np.linalg.norm(carpose,axis=0)
    if debug and (i%10)==0:
        print(carpose)
        print(transformnorms[0:3])
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        mngr.window.wm_geometry("+600+400")
        xmax = 0.25
        xmin = -xmax
        fig1 = plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(imcv, cv2.COLOR_BGR2RGB))


        fig2 = plt.subplot(1, 3, 2)
        plt.title("Global Coordinates")
        plt.scatter(positionsglobal[:,0], positionsglobal[:,1], label='measured data', facecolors='none', edgecolors='b', s=8)
        plt.plot(labelposesglobal[:,0,3], labelposesglobal[:,1,3], label='samples from spline fit', c='r')
        plt.plot(carpose[0,3], carpose[1,3], 'g*', label='current car position')

        
        plt.arrow(carpose[0,3], carpose[1,3], 0.25*carpose[0,0], 0.25*carpose[1,0])
        fig3 = plt.subplot(1, 3, 3)
        plt.title("Local Coordinates")
        plt.scatter(-positionslocal[:,1], positionslocal[:,0], label='measured data', facecolors='none', edgecolors='b', s=8)
        plt.plot(-labelpositions[:,1], labelpositions[:,0], label='samples from spline fit', c='r')
        plt.plot(labelposes[0,1,3], labelposes[0,0,3], 'g*', label='current car position')
        fig3.set_xlim(xmin,xmax)
       # plt.arrow(carpose[0,3], carpose[1,3], 0.25*carpose[0,1], 0.25*carpose[1,1])
        #WX = np.
        #plt.quiver(X, Y, dX, dY, color='r')
        #plt.quiver([], [], [], [])
        plt.show()
        print(label_tag)
    
    assert(np.allclose(transformnorms[0:3],np.ones(3)))
    pointsgood=True
    for j in range(sample_indices):
        pose_forward_pb = Pose3d_pb2.Pose3d()
        newpose = label_tag.subsequent_poses.add()
        newvel = label_tag.subsequent_linear_velocities.add()
        newangvel = label_tag.subsequent_angular_velocities.add()
        newpose.frame = FrameId_pb2.LOCAL
        newvel.frame = FrameId_pb2.LOCAL
        newangvel.frame = FrameId_pb2.LOCAL
        if(labelpositions[j,0]<-0.005):
            #raise ValueError("Somehow got a negative forward value on image %d. " %(i,))
            print("Somehow got a negative forward value on image %d. Skipping that image" %(i,))
            pointsgood=False
            break
        newpose.translation.CopyFrom(proto_utils.vectorFromNumpy(labelpositions[j]))
        newpose.rotation.CopyFrom(proto_utils.quaternionFromNumpy(labelquats[j]))

        newvel.vector.CopyFrom(proto_utils.vectorFromNumpy(labelvels[j]))

        newangvel.vector.CopyFrom(proto_utils.vectorFromNumpy(labelangvels[j]))

        #labeltime = imagetimes[i+j]
        #labeltime = odomtimes[iclosest+j]
        labeltime = tsamp[j]
       # tsamp
        newpose.session_time = labeltime
        newvel.session_time = labeltime
        newangvel.session_time = labeltime
    if not pointsgood:
        continue
    label_tag_file_path = os.path.join(labeldir, key + "_sequence_label.json")
    with open(label_tag_file_path,'w') as f:
        f.write(google.protobuf.json_format.MessageToJson(label_tag, including_default_value_fields=True))
    label_tag_file_path_binary = os.path.join(labeldir, key + "_sequence_label.pb")
    with open(label_tag_file_path_binary,'wb') as f:
        f.write( label_tag.SerializeToString() )
    
    
    
    db.writePoseSequenceLabel(key, label_tag)
videowriter.release()
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