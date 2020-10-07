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
import cv2
import shutil
import os
import time
import deepracing, deepracing.backend, deepracing.protobuf_utils as proto_utils
import google.protobuf.json_format
import bisect
import PIL
from PIL import Image, ImageFilter, ImageDraw
import json
from scipy.spatial.kdtree import KDTree
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
parser.add_argument('--raceline', type=str, required=True , help="Path to the raceline json to read")
parser.add_argument('--lookahead_distance', type=float, default=1.5, help="Look ahead this many meters from the ego pose")
parser.add_argument('--num_samples', type=int, default=60, help="How many points to sample along the lookahead distance")
parser.add_argument('--debug', action="store_true", help="Display some debug plots")
parser.add_argument('--mintime', type=float, default=5.0, help="Ignore this many seconds of data from the beginning of the bag file")
parser.add_argument('--maxtime', type=float, default=7.5, help="Ignore this many seconds of leading up to the end of the bag file")
args = parser.parse_args()
argdict = vars(args)
lookahead_distance = argdict["lookahead_distance"]
num_samples = argdict["num_samples"]
configfile = argdict["config"]
bagpath = argdict["bagfile"]
mintime = argdict["mintime"]
maxtime = argdict["maxtime"]
debug = argdict["debug"]
racelinefile = argdict["raceline"]
bagdir = os.path.dirname(bagpath)
bridge = cv_bridge.CvBridge()
if configfile is None:
    configfile = os.path.join(bagdir,"topicconfig.yaml")
with open(configfile,'r') as f:
    topicdict : dict =yaml.load(f, Loader=yaml.SafeLoader)
with open(racelinefile,'r') as f:
    racelinedict : dict = json.load(f)
raceline = np.column_stack([np.array(racelinedict["x"], dtype=np.float64).copy(), np.array(racelinedict["y"], dtype=np.float64).copy(), np.array(racelinedict["z"], dtype=np.float64).copy()])
racelinedist = np.array(racelinedict["dist"], dtype=np.float64).copy()
print("racelinedist.shape: %s" % (str(racelinedist.shape),))
bezier_order = racelinedict["bezier_order"]
racelinekdtree = KDTree(raceline.copy())
bag = rosbag.Bag(bagpath)
msg_types, typedict = bag.get_type_and_topic_info()
print(typedict)
typeandtopicinfo = bag.get_type_and_topic_info()
topics = typeandtopicinfo.topics
msgdict = {topic: [] for topic in topics.keys()}
for topic, msg, t in tqdm(iterable=bag.read_messages(topics=None), desc="Loading messages from bag file", total=bag.get_message_count()):
    msgdict[topic].append(msg)

odoms = sorted(msgdict[topicdict["odom"]], key=sortKey)
print(len(odoms))
images = sorted(msgdict[topicdict["images"]], key=sortKey)
imagetimes = np.array([stampToSeconds(i.header.stamp) for i in images], dtype=np.float64)
odomtimes = np.array([stampToSeconds(o.header.stamp) for o in odoms], dtype=np.float64)
t0 = imagetimes[0]
imagetimes = imagetimes - t0
odomtimes = odomtimes - t0
tbuff = int(np.round(0.5/np.mean(odomtimes[1:] - odomtimes[0:-1])))
meanrldist = np.mean(racelinedist[1:] - racelinedist[:-1])
racelinebuff = int(np.round(lookahead_distance/meanrldist))
print("racelinebuff: %d" % (racelinebuff,))

print(imagetimes)
print(odomtimes)
posemsgs = [o.pose.pose for o in odoms]
positions = np.array([ [p.position.x, p.position.y, p.position.z] for p in posemsgs], dtype=np.float64)
quaternions = np.array([ [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w] for p in posemsgs], dtype=np.float64)

k = 5
nspline = 60
print("Writing labels to file")
rootdir = os.path.join(bagdir, os.path.splitext(os.path.basename(bagpath))[0])
if os.path.isdir(rootdir):
    shutil.rmtree(rootdir, ignore_errors=True)
    time.sleep(0.5)
imagedir = os.path.join(rootdir,"images")
imagelmdbdir = os.path.join(imagedir,"image_lmdb")
labeldir = os.path.join(rootdir,"raceline_labels")
labellmdbdir = os.path.join(labeldir,"lmdb")
os.makedirs(imagedir)
os.makedirs(labeldir)
os.makedirs(imagelmdbdir)
os.makedirs(labellmdbdir)
imagebackend = deepracing.backend.ImageLMDBWrapper()
imagebackend.readDatabase(imagelmdbdir,mapsize=int(round(1.5*len(images)*66*200*3)), readonly=False)
labelbackend = deepracing.backend.MultiAgentLabelLMDBWrapper()
labelbackend.openDatabase(labellmdbdir, readonly=False)
goodkeys = []
up = np.array([0.0,0.0,1.0], dtype=np.float64)
trate = 1.0/15.0
for (i, timage) in tqdm(enumerate(imagetimes), total=len(imagetimes)):
    tick = time.time()
    imagetag = TimestampedImage_pb2.TimestampedImage()
    imageprefix =  "image_%d"
    imagetag.image_file = (imageprefix +".jpg") % i
    imagetag.timestamp = timage
    imnp = bridge.compressed_imgmsg_to_cv2(images[i], desired_encoding="rgb8")
    impil = PIL.Image.fromarray(imnp)
    impil.save(os.path.join(imagedir, imagetag.image_file))
    rows = imnp.shape[0]
    cols = imnp.shape[1]
    rowstart = int(round(0.55*rows))
    imcropped = impil.crop([0,rowstart, cols-1, rows-1])
    impilresize = imagebackend.writeImage(imageprefix%i, imcropped)
    imnpresize = np.array(impilresize)

    with open(os.path.join(imagedir, (imageprefix +".json") % i), "w") as f:
        f.write(google.protobuf.json_format.MessageToJson(imagetag, including_default_value_fields=True, indent=2))
    with open(os.path.join(imagedir, (imageprefix +".pb") % i), "wb") as f:
        f.write(imagetag.SerializeToString())

        
    if timage<=mintime or timage>=imagetimes[-1]-maxtime:
        continue
    labeltag = MultiAgentLabel_pb2.MultiAgentLabel()
    labeltag.image_tag.CopyFrom(imagetag)

    isamp = bisect.bisect_left(odomtimes, timage)
    tfit = odomtimes[isamp-tbuff:isamp+nspline+tbuff]
    pfit = positions[isamp-tbuff:isamp+nspline+tbuff]
    
    spl : scipy.interpolate.LSQUnivariateSpline = scipy.interpolate.make_lsq_spline(tfit, pfit, sensibleKnots(tfit, k), k=k)
    splvel : scipy.interpolate.LSQUnivariateSpline = spl.derivative()
    tsamp = np.linspace(timage, timage+1.5, num = num_samples)
    splvals = spl(tsamp)
    splvelvals = splvel(tsamp)
    
    cartraj = np.stack( [np.eye(4,dtype=np.float64) for asdf in range(tsamp.shape[0])], axis=0)
    for j in range(cartraj.shape[0]):
        posspl = splvals[j]
        velspl = splvelvals[j]
        rx = velspl/np.linalg.norm(velspl, ord=2)
        ry = np.cross(up,rx)
        ry = ry/np.linalg.norm(ry, ord=2)
        rz = np.cross(rx,ry)
        rz = rz/np.linalg.norm(rz, ord=2)
        cartraj[j,0:3,3] = posspl
        cartraj[j,0:3,0:3] = np.column_stack([rx,ry,rz])
    #print(rz)
    
    carpose = cartraj[0].copy()
    carrotation = Rot.from_matrix(carpose[0:3,0:3].copy())
    carposeinv = np.linalg.inv(carpose)

    labeltag.ego_agent_pose.translation.CopyFrom(proto_utils.vectorFromNumpy(carpose[0:3,3]))
    labeltag.ego_agent_pose.rotation.CopyFrom(proto_utils.quaternionFromScipy(carrotation))
    labeltag.ego_agent_pose.frame = FrameId_pb2.GLOBAL
    labeltag.ego_agent_pose.session_time = tsamp[0]

    labeltag.ego_agent_linear_velocity.vector.CopyFrom(proto_utils.vectorFromNumpy(splvelvals[0]))
    labeltag.ego_agent_linear_velocity.session_time = tsamp[0]
    labeltag.ego_agent_linear_velocity.frame = FrameId_pb2.GLOBAL

    labeltag.raceline_frame = FrameId_pb2.LOCAL



    _, iclosest = racelinekdtree.query(carpose[0:3,3])
    rlidx = np.arange(iclosest-int(round(racelinebuff/8)), iclosest+racelinebuff+1,step=1, dtype=np.int64)%raceline.shape[0]

    rld = racelinedist[rlidx]
  #  print(rld)
    overlapidx = rld<rld[0]
    irldmax = np.argmax(rld)
  #  print(overlapidx)
    rld[overlapidx]+=rld[irldmax] + meanrldist
   # print(rld)

    rlglobal = raceline[rlidx]
    rlspline = scipy.interpolate.make_lsq_spline(rld, rlglobal, sensibleKnots(rld,k), k=k)
    rldsamp = np.linspace(rld[0], rld[-1], num=num_samples)
    labeltag.ClearField("raceline_distances")
    labeltag.raceline_distances.extend(rldsamp.tolist())

    rlsampglobal = rlspline(rldsamp)
    rlsamplocal = np.matmul(carposeinv, np.row_stack([rlsampglobal.transpose(), np.ones_like(rlsampglobal[:,0])]))[0:3].transpose()
    cartrajlocal = np.matmul(carposeinv, cartraj)
    for j in range(num_samples):
        newvec = labeltag.local_raceline.add()
        newvec.CopyFrom(proto_utils.vectorFromNumpy(rlsamplocal[j]))

        newpose = labeltag.ego_agent_trajectory.poses.add()
        newpose.frame = FrameId_pb2.LOCAL
        newpose.translation.CopyFrom(proto_utils.vectorFromNumpy(cartrajlocal[j,0:3,3]))
        newpose.rotation.CopyFrom(proto_utils.quaternionFromScipy(Rot.from_matrix(cartrajlocal[j,0:3,0:3])))
        newpose.session_time = tsamp[j]
    labeltag.ego_car_index = 0
    labeltag.track_id=26
    with open(os.path.join(labeldir, (imageprefix +".json") % i), "w") as f:
        f.write(google.protobuf.json_format.MessageToJson(labeltag, including_default_value_fields=True, indent=2))
    with open(os.path.join(labeldir, (imageprefix +".pb") % i), "wb") as f:
        f.write(labeltag.SerializeToString())
    labelbackend.writeMultiAgentLabel(imageprefix%i, labeltag)
    goodkeys.append((imageprefix%i)+"\n")
    tock = time.time()
    dt = (tock-tick)
    if debug:
        key = goodkeys[-1].replace("\n","")
        imnpdb = imagebackend.getImage(key)
        lbldb = labelbackend.getMultiAgentLabel(key)
        lbldbrl = lbldb.local_raceline
        racelinelocal =  np.array([[p.x,  p.y, p.z, 1.0 ]  for p in lbldbrl ], dtype=np.float64 ).transpose()
        egopose = np.eye(4,dtype=np.float64)
        egopose[0:3,3] = np.array([lbldb.ego_agent_pose.translation.x, lbldb.ego_agent_pose.translation.y, lbldb.ego_agent_pose.translation.z ], dtype=np.float64)
        egopose[0:3,0:3] = Rot.from_quat(np.array([lbldb.ego_agent_pose.rotation.x, lbldb.ego_agent_pose.rotation.y, lbldb.ego_agent_pose.rotation.z, lbldb.ego_agent_pose.rotation.w], dtype=np.float64)).as_matrix()
        egotrajpb = lbldb.ego_agent_trajectory
        egotrajlocal = np.array([[p.translation.x,  p.translation.y, p.translation.z, 1.0 ]  for p in egotrajpb.poses ], dtype=np.float64 ).transpose()
        egotrajglobal = np.matmul(egopose, egotrajlocal)
        racelineglobal = np.matmul(egopose, racelinelocal)
        fig1 = plt.subplot(1, 3, 1)
        plt.imshow(imnpdb)
        plt.title("Image %d" % i)
        fig2 = plt.subplot(1, 3, 2)
        plt.title("Global Coordinates")
        plt.scatter(pfit[:,0], pfit[:,1], label="Data", facecolors="none", edgecolors="blue")
        plt.plot(egotrajglobal[0], egotrajglobal[1], label="Ego Agent Trajectory Label", c="r")
        plt.plot(racelineglobal[0], racelineglobal[1], label="Optimal Raceline", c="g")
        plt.plot(egotrajglobal[0,0], egotrajglobal[1,0], "g*", label="Position of Car")
        fig3 = plt.subplot(1, 3, 3)
        plt.title("Local Coordinates")
        plt.plot(-egotrajlocal[1], egotrajlocal[0], label="Ego Agent Trajectory Label", c="r")
        plt.plot(-racelinelocal[1], racelinelocal[0], label="Optimal Raceline", c="g")
        plt.plot(-egotrajlocal[1,0], egotrajlocal[0,0], "g*", label="Position of Car")
      #  plt.arrow(splvals[0,0], splvals[0,1], rx[0], rx[1], label="Velocity of Car")
        plt.show()
    if dt<trate:
        time.sleep(trate - dt)
with open(os.path.join(rootdir,"goodkeys.txt"),"w") as f:
    f.writelines(goodkeys)