import rosbag
import yaml
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan, LaserEcho, Image, CompressedImage
import argparse
import TimestampedImage_pb2, TimestampedPacketMotionData_pb2, PoseSequenceLabel_pb2, Image_pb2, FrameId_pb2, MultiAgentLabel_pb2
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
import deepracing_models, deepracing_models.math_utils as math_utils
import google.protobuf.json_format
import bisect
import PIL
from PIL import Image, ImageFilter, ImageDraw
import torch
import json
parser = argparse.ArgumentParser()
parser.add_argument("racelinebag", help="Path to trackfile to convert",  type=str)
parser.add_argument("num_samples", type=int, help="Number of values to sample from the spline. Default (0) means no sampling and just copy the data as is")
parser.add_argument("--topic", default="overtake", type=str, help="which topic to get raceline from")
parser.add_argument("--k", default=1, type=int, help="Order of bezier curve to use for the fit")
#parser.add_argument("--negate_normals", action="store_true", help="Flip the sign all all of the computed normal vectors")
args = parser.parse_args()
argdict = vars(args)
racelinebag = argdict["racelinebag"]
num_samples = argdict["num_samples"]
k = argdict["k"]


bagdir = os.path.dirname(racelinebag)
bag = rosbag.Bag(racelinebag)
typeandtopicinfo = bag.get_type_and_topic_info()
topics = typeandtopicinfo.topics
msgdict = {topic: [] for topic in topics.keys()}
print(topics)
rltopic = "/trajectory/raceline"
ottopic = "/trajectory/overtake"
subtopic = argdict["topic"]
chosentopic = "/trajectory/%s" % subtopic
print(rltopic)
for topic, msg, t in tqdm(iterable=bag.read_messages(topics=None), desc="Loading messages from bag file", total=bag.get_message_count()):
    msgdict[topic].append(msg)
raceline : Path = msgdict[rltopic][0]
overtake : Path = msgdict[ottopic][0]
chosenpath : Path = msgdict[chosentopic][0]

rlnp = np.array([  [p.pose.position.x, p.pose.position.y, p.pose.position.z ] for p in raceline.poses ], dtype=np.float64)[:-1]
otnp = np.array([  [p.pose.position.x, p.pose.position.y, p.pose.position.z ] for p in overtake.poses ], dtype=np.float64)[:-1]

chosenpathnp = np.array([  [p.pose.position.x, p.pose.position.y, p.pose.position.z ] for p in chosenpath.poses ], dtype=np.float64)[:-1]
chosenpathdists = np.hstack([np.zeros(1), np.cumsum(np.linalg.norm(chosenpathnp[1:]-chosenpathnp[0:-1], ord=2, axis=1))])
print(chosenpathdists)

spl : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(chosenpathdists, chosenpathnp, k=k)
tanspl : scipy.interpolate.BSpline = spl.derivative()
dsamp = np.linspace(chosenpathdists[0], chosenpathdists[-1], num=num_samples)
chosenpatheval = spl(dsamp)
chosenpathtangents = tanspl(dsamp)
up = np.row_stack( [np.array([0.0,0.0,1.0], dtype=np.float64) for asdf in range(dsamp.shape[0])] )
inward = np.cross(up,chosenpathtangents)
inward = inward/np.linalg.norm(inward,ord=2,axis=1)[:,np.newaxis]

chosenpatheval = chosenpatheval + 0.1*inward
dsamp = np.hstack([np.zeros(1), np.cumsum(np.linalg.norm(chosenpatheval[1:]-chosenpatheval[0:-1], ord=2, axis=1))])
print(dsamp)


fig = plt.figure()
plt.scatter(rlnp[:,0], rlnp[:,1], edgecolor="b", facecolor="none", label="Raceline Path on subtopic %s" % subtopic)
plt.scatter(otnp[:,0], otnp[:,1], edgecolor="b", facecolor="none", label="Overtake Path on subtopic %s" % subtopic)
plt.plot(chosenpatheval[:,0], chosenpatheval[:,1], label="best fit bc", c="g")
plt.plot(chosenpatheval[0,0], chosenpatheval[0,1], "g*", label="Initial Point")
plt.show()


rldict = {"x": chosenpatheval[:,0].tolist(), "y": chosenpatheval[:,1].tolist(), "z": chosenpatheval[:,2].tolist(), "dist": dsamp.tolist(), "splk" : k} 

with open(os.path.join(bagdir, "%s.json" % subtopic),"w") as f:
    json.dump(rldict, f, indent=2)