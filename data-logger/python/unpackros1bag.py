import rospkg 
import os
if "add_dll_directory" in dir(os):
    rospack = rospkg.RosPack()
    rosbinpaths = set([os.path.abspath(os.path.join(rospack.get_path(pkgname),os.pardir,os.pardir,"bin")) for pkgname in ["cv_bridge", "rosbag"] ])
    for path in rosbinpaths:
        os.add_dll_directory(path)
    vcpkg_bin_dir = os.getenv("VCPKG_BIN_DIR")
    if vcpkg_bin_dir is not None:
        os.add_dll_directory(vcpkg_bin_dir)
import rosbag
import yaml
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, LaserEcho, Image, CompressedImage
import argparse
import TimestampedImage_pb2, TimestampedPacketMotionData_pb2, Image_pb2, FrameId_pb2, MultiAgentLabel_pb2
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
import shutil
import time
import deepracing, deepracing.backend, deepracing.protobuf_utils as proto_utils
import google.protobuf.json_format
import bisect
import PIL
from PIL import Image, ImageFilter, ImageDraw
import json
from scipy.spatial.kdtree import KDTree
from copy import deepcopy
import torch
import torchvision.transforms.functional as F
# /car_1/camera/image_raw      8263 msgs    : sensor_msgs/Image
# /car_1/multiplexer/command   7495 msgs    : ackermann_msgs/AckermannDrive
# /car_1/odom_filtered         6950 msgs    : nav_msgs/Odometry
# /car_1/scan                  9870 msgs    : sensor_msgs/LaserScan
def sensibleKnots(t, degree):
    numsamples = t.shape[0]
    knots = [ t[int(numsamples/4)], t[int(numsamples/2)], t[int(3*numsamples/4)] ]
    knots = np.r_[(t[0],)*(degree+1),  knots,  (t[-1],)*(degree+1)]
    return knots
def stampToTime(stamp):
    return Time(secs=stamp.secs, nsecs=stamp.nsecs)
def stampToSeconds(stamp):
    return stampToTime(stamp).to_sec()
def sortKey(msg):
    return stampToSeconds(msg.header.stamp)

parser = argparse.ArgumentParser("Unpack a bag file into a dataset")
parser.add_argument('bagfile', type=str,  help='The bagfile to unpack')
parser.add_argument('--config', type=str, required=False, default=None , help="Config file specifying the rostopics to unpack, defaults to a file named \"topicconfig.yaml\" in the same directory as the bagfile")
parser.add_argument('--lookahead_time', type=float, default=1.5, help="Look ahead this seconds for the ego trajectory labels")
parser.add_argument('--num_samples', type=int, default=120, help="How many points to sample along the lookahead distance")
parser.add_argument('--k', type=int, default=3, help="Order of the least squares splines to fit to the noisy data")
parser.add_argument('--debug', action="store_true", help="Display some debug plots")
parser.add_argument('--mintime', type=float, default=5.0, help="Ignore this many seconds of data from the beginning of the bag file")
parser.add_argument('--maxtime', type=float, default=7.5, help="Ignore this many seconds of leading up to the end of the bag file")
parser.add_argument('--rowstart', type=float, default=0.5, help="Ratio to crop off the top of the image")
parser.add_argument('--labeldir', type=str, required=False, default="pose_sequence_labels" , help="Where to put the labels relative to the bagfile")


args = parser.parse_args()
argdict = vars(args)
labeldirarg = argdict["labeldir"]
lookahead_time = argdict["lookahead_time"]
num_samples = argdict["num_samples"]
configfile = argdict["config"]
bagpath = argdict["bagfile"]
mintime = argdict["mintime"]
maxtime = argdict["maxtime"]
debug = argdict["debug"]
rowstart = argdict["rowstart"]
k = argdict["k"]
bagdir = os.path.dirname(bagpath)

rootdir = os.path.join(bagdir, os.path.splitext(os.path.basename(bagpath))[0])
imagedir = os.path.join(rootdir,"images")
croppedimagedir = os.path.join(rootdir,"cropped_images")
imagelmdbdir = os.path.join(imagedir,"image_lmdb")
labeldir = os.path.join(rootdir,labeldirarg)
labellmdbdir = os.path.join(labeldir,"lmdb")
os.makedirs(labellmdbdir, exist_ok=False)

os.makedirs(imagedir, exist_ok=True)
os.makedirs(croppedimagedir, exist_ok=True)
os.makedirs(imagelmdbdir, exist_ok=True)



bridge = cv_bridge.CvBridge()
if configfile is None:
    configfile = os.path.join(bagdir,"topicconfig.yaml")
with open(configfile,'r') as f:
    topicdict : dict =yaml.load(f, Loader=yaml.SafeLoader)
bag = rosbag.Bag(bagpath)
msg_types, typedict = bag.get_type_and_topic_info()
print(typedict)
typeandtopicinfo = bag.get_type_and_topic_info()
topics = typeandtopicinfo.topics
msgdict = {topic: [] for topic in topics.keys()}
for topic, msg, t in tqdm(iterable=bag.read_messages(topics=None), desc="Loading messages from bag file", total=bag.get_message_count()):
    msgdict[topic].append(msg)

odoms = sorted(msgdict[topicdict["odom"]], key=sortKey)

images = sorted(msgdict[topicdict["images"]], key=sortKey)

laserscans = sorted(msgdict[topicdict["scan"]], key=sortKey)

#print(laserscans[0])

imagetimes = np.array([stampToSeconds(i.header.stamp) for i in images], dtype=np.float64)
odomtimes = np.array([stampToSeconds(o.header.stamp) for o in odoms], dtype=np.float64)
laserscantimes = np.array([stampToSeconds(ls.header.stamp) for ls in laserscans], dtype=np.float64)


t0 = deepcopy(imagetimes[0])
imagetimes = imagetimes - t0
odomtimes = odomtimes - t0
laserscantimes = laserscantimes - t0





dtindices = int(round(lookahead_time/np.mean(odomtimes[1:] - odomtimes[0:-1])))
dtbuff = int(round(0.5*dtindices))

posemsgs = [o.pose.pose for o in odoms]
positions = np.array([ [p.position.x, p.position.y, p.position.z] for p in posemsgs], dtype=np.float64)
quaternions = np.array([ [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w] for p in posemsgs], dtype=np.float64)

try:
    imnp0 = bridge.imgmsg_to_cv2(images[0], desired_encoding="rgb8")
except Exception as e:
    imnp0 = bridge.compressed_imgmsg_to_cv2(images[0], desired_encoding="rgb8")
croprow = int(round(imnp0.shape[0]*rowstart))
imsize = np.array([66,200])
imagebackend = deepracing.backend.ImageLMDBWrapper()
imagebackend.readDatabase(imagelmdbdir,mapsize=int(round(1.1*len(images)*3*np.prod(imsize))), readonly=False)
labelbackend = deepracing.backend.MultiAgentLabelLMDBWrapper()
labelbackend.openDatabase(labellmdbdir, mapsize=30000*len(images), readonly=False)
goodkeys = []
up = np.array([0.0,0.0,1.0], dtype=np.float64)
trate = 1.0/15.0

imageconfig = {k: argdict[k] for k in ['rowstart']}
imageconfig['inputsize'] = [imnp0.shape[0], imnp0.shape[1]]
imageconfig['outputsize'] = imsize.tolist()
with open(os.path.join(imagedir,"config.json"),"w") as f:
    json.dump(imageconfig, f, indent=2)

labelconfig = {k: argdict[k] for k in ['mintime', 'maxtime', 'lookahead_time', 'num_samples']}
with open(os.path.join(labeldir,"config.json"),"w") as f:
    json.dump(labelconfig, f, indent=2)

print("Writing labels to file")
try:
    for (i, timage) in tqdm(enumerate(imagetimes), total=len(imagetimes)):
        tick = time.time()
        imagetag = TimestampedImage_pb2.TimestampedImage()
        key =  "image_%d" % i
        imagetag.image_file = key+".jpg"
        imagetag.timestamp = timage
        try:
            impil = PIL.Image.fromarray(bridge.imgmsg_to_cv2(images[i], desired_encoding="rgb8"))
        except Exception as e:
            impil = PIL.Image.fromarray(bridge.compressed_imgmsg_to_cv2(images[i], desired_encoding="rgb8"))
        impil.save(os.path.join(imagedir, imagetag.image_file))
        imcropped : PIL.Image.Image = impil.crop([0, croprow, impil.width, impil.height])
        imcropped.save(os.path.join(croppedimagedir, imagetag.image_file))
        imlmdb : PIL.Image.Image = F.resize(imcropped, imsize, interpolation = PIL.Image.LANCZOS)
        impb = imagebackend.writeImage(key, imlmdb)

        with open(os.path.join(imagedir, key +".json"), "w") as f:
            f.write(google.protobuf.json_format.MessageToJson(imagetag, including_default_value_fields=True, indent=2))
            
        if timage<=mintime or timage>=imagetimes[-1]-maxtime:
            continue
        labeltag = MultiAgentLabel_pb2.MultiAgentLabel()
        labeltag.image_tag.CopyFrom(imagetag)

        tsamp = np.linspace(timage, timage + lookahead_time, num = num_samples)
        istart = bisect.bisect_left(odomtimes, tsamp[0])
        iend = bisect.bisect_left(odomtimes, tsamp[-1])
        tfit = odomtimes[istart-dtbuff:iend+dtbuff]
        pfit = positions[istart-dtbuff:iend+dtbuff]
        qfit = quaternions[istart-dtbuff:iend+dtbuff]
        
        spl : scipy.interpolate.LSQUnivariateSpline = scipy.interpolate.make_lsq_spline(tfit, pfit, sensibleKnots(tfit, k), k=k)
        splder : scipy.interpolate.LSQUnivariateSpline = spl.derivative()
        
        cartraj = np.stack( [np.eye(4,dtype=np.float64) for asdf in range(tsamp.shape[0])], axis=0)
        cartraj[:,0:3,3] = spl(tsamp)
        linearvelsglobal = splder(tsamp)
        for j in range(cartraj.shape[0]):
            velspl = linearvelsglobal[j]
            rx = velspl/np.linalg.norm(velspl, ord=2)
            ry = np.cross(up,rx)
            ry = ry/np.linalg.norm(ry, ord=2)
            rz = np.cross(rx,ry)
            rz = rz/np.linalg.norm(rz, ord=2)
            cartraj[j,0:3,0:3] = np.column_stack([rx,ry,rz])
           # q = r.as_quat()
           # print("Distance between rotations: %f" % np.arccos(-1+ 2*(np.dot(q,qsamp[j])**2)))
            #cartraj[j,0:3,0:3] = np.column_stack([rx,ry,rz])
      #  print(qsamp)
        carrotations = Rot.from_matrix(cartraj[:,0:3,0:3])
        
        carrotationspline = RotSpline(tsamp,carrotations)
        angvelsglobal = carrotationspline(tsamp,1)
        #print(rz)
        
        carpose = cartraj[0].copy()
        carposeinv = np.linalg.inv(carpose)

        labeltag.ego_agent_pose.translation.CopyFrom(proto_utils.vectorFromNumpy(carpose[0:3,3]))
        labeltag.ego_agent_pose.rotation.CopyFrom(proto_utils.quaternionFromScipy(carrotations[0]))
        labeltag.ego_agent_pose.frame = FrameId_pb2.GLOBAL
        labeltag.ego_agent_pose.session_time = tsamp[0]

        labeltag.ego_agent_linear_velocity.vector.CopyFrom(proto_utils.vectorFromNumpy(linearvelsglobal[0]))
        labeltag.ego_agent_linear_velocity.frame = FrameId_pb2.GLOBAL
        labeltag.ego_agent_linear_velocity.session_time = tsamp[0]

        labeltag.ego_agent_angular_velocity.vector.CopyFrom(proto_utils.vectorFromNumpy(angvelsglobal[0]))
        labeltag.ego_agent_linear_velocity.frame = FrameId_pb2.GLOBAL
        labeltag.ego_agent_angular_velocity.session_time = tsamp[0]

        cartrajlocal = np.matmul(carposeinv, cartraj)
        linearvelslocal = np.matmul(carposeinv[0:3,0:3], linearvelsglobal.transpose()).transpose()
        angvelslocal = np.matmul(carposeinv[0:3,0:3], angvelsglobal.transpose()).transpose()
        for j in range(num_samples):

            newpose = labeltag.ego_agent_trajectory.poses.add()
            newpose.frame = FrameId_pb2.LOCAL
            newpose.translation.CopyFrom(proto_utils.vectorFromNumpy(cartrajlocal[j,0:3,3]))
            newpose.rotation.CopyFrom(proto_utils.quaternionFromScipy(Rot.from_matrix(cartrajlocal[j,0:3,0:3])))
            newpose.session_time = tsamp[j]

            newlinearvel = labeltag.ego_agent_trajectory.linear_velocities.add()
            newlinearvel.frame = FrameId_pb2.LOCAL
            newlinearvel.vector.CopyFrom(proto_utils.vectorFromNumpy(linearvelslocal[j]))
            newlinearvel.session_time = tsamp[j]

            newangularvel = labeltag.ego_agent_trajectory.angular_velocities.add()
            newangularvel.frame = FrameId_pb2.LOCAL
            newangularvel.vector.CopyFrom(proto_utils.vectorFromNumpy(angvelslocal[j]))
            newangularvel.session_time = tsamp[j]

        labeltag.ego_car_index = 0
        labeltag.track_id=26
        with open(os.path.join(labeldir, key+".json"), "w")  as f:
            f.write(google.protobuf.json_format.MessageToJson(labeltag, including_default_value_fields=True, indent=2))
        with open(os.path.join(labeldir, key+".pb"), "wb")  as f:
            f.write(labeltag.SerializeToString())

        labelbackend.writeMultiAgentLabel(key, labeltag)
        goodkeys.append(key)
        tock = time.time()
        dt = (tock-tick)
        if debug and i%5==0:
            key = goodkeys[-1].replace("\n","")

            imnp = np.asarray(impil)
            imnpdb = imagebackend.getImage(key)

            lbldb = labelbackend.getMultiAgentLabel(key)

            egopose = np.eye(4,dtype=np.float64)
            egopose[0:3,3] = np.array([ lbldb.ego_agent_pose.translation.x, lbldb.ego_agent_pose.translation.y, lbldb.ego_agent_pose.translation.z ], dtype=np.float64)
            egopose[0:3,0:3] = Rot.from_quat(np.array([ lbldb.ego_agent_pose.rotation.x, lbldb.ego_agent_pose.rotation.y, lbldb.ego_agent_pose.rotation.z, lbldb.ego_agent_pose.rotation.w ], dtype=np.float64)).as_matrix()
            egotrajpb = lbldb.ego_agent_trajectory
            egotrajlocal = np.array([ [p.translation.x,  p.translation.y, p.translation.z, 1.0 ]  for p in egotrajpb.poses ], dtype=np.float64 ).transpose()
            pfitlocal = np.matmul(np.linalg.inv(egopose), np.row_stack([pfit.transpose(), np.ones_like(pfit[:,0])]))[0:3].transpose()
            egotrajglobal = np.matmul(egopose, egotrajlocal)

            fig1 = plt.subplot(1, 3, 1)
            plt.imshow(imnpdb)
            plt.title("Image %d" % i)
            fig2 = plt.subplot(1, 3, 2)
            plt.title("Global Coordinates")
            plt.scatter(pfit[:,0], pfit[:,1], label="PF Estimates", facecolors="none", edgecolors="blue")
            plt.plot(egotrajglobal[0], egotrajglobal[1], label="Ego Agent Trajectory Label", c="r")
            plt.plot(egotrajglobal[0,0], egotrajglobal[1,0], "g*", label="Position of Car")
            fig3 = plt.subplot(1, 3, 3)
            plt.title("Local Coordinates")
            plt.scatter(pfitlocal[:,1], pfitlocal[:,0], label="PF Estimates", facecolors="none", edgecolors="blue")
            plt.plot(egotrajlocal[1], egotrajlocal[0], label="Ego Agent Trajectory Label", c="r")
            plt.plot(egotrajlocal[1,0], egotrajlocal[0,0], "g*", label="Position of Car")
            plt.plot([0.0, 0.0], [0.0, 1.5], label="Forward", c="black")
            xmin = np.min(egotrajlocal[1]) - 0.05
            xmax = np.max(egotrajlocal[1]) + 0.05
            plt.xlim(xmax,xmin)
            plt.legend()
        #  plt.arrow(splvals[0,0], splvals[0,1], rx[0], rx[1], label="Velocity of Car")
            plt.show()
        # if dt<trate:
        #     time.sleep(trate - dt)
except KeyboardInterrupt as e:
    pass
with open(os.path.join(rootdir,"goodkeys.txt"),"w") as f:
    f.writelines([k+"\n" for k in goodkeys])