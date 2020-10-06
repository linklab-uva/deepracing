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
parser.add_argument("--k", default=31, type=int, help="Order of bezier curve to use for the fit")
#parser.add_argument("--negate_normals", action="store_true", help="Flip the sign all all of the computed normal vectors")
args = parser.parse_args()
argdict = vars(args)
racelinebag = argdict["racelinebag"]
num_samples = argdict["num_samples"]
bezierorder = argdict["k"]


bagdir = os.path.dirname(racelinebag)
bag = rosbag.Bag(racelinebag)
typeandtopicinfo = bag.get_type_and_topic_info()
topics = typeandtopicinfo.topics
msgdict = {topic: [] for topic in topics.keys()}
print(topics)
subtopic = argdict["topic"]
rltopic = "/trajectory/%s" % subtopic
print(rltopic)
for topic, msg, t in tqdm(iterable=bag.read_messages(topics=None), desc="Loading messages from bag file", total=bag.get_message_count()):
    msgdict[topic].append(msg)
raceline : Path = msgdict[rltopic][0]
rlnp = np.array([  [p.pose.position.x, p.pose.position.y, p.pose.position.z ] for p in raceline.poses ], dtype=np.float64)
rlnp = rlnp[0:-1]
rldiffs = rlnp[1:] - rlnp[0:-1]
rldiffnorms = np.linalg.norm(rldiffs, ord=2, axis=1)
rl = torch.from_numpy(rlnp.copy())
rldistances = torch.from_numpy((np.concatenate([np.zeros(1,dtype=np.float64), np.cumsum(rldiffnorms)])).copy())
# print(rl.shape)
# print(rldistances)
dmax = rldistances[-1]
s = rldistances/dmax
# print(s)




Mlstsq, bc = math_utils.bezierLsqfit(rl.unsqueeze(0), bezierorder, t=s.unsqueeze(0))

ssamp = torch.linspace(0.0,1.0,steps=num_samples, dtype=torch.float64).unsqueeze(0)
dsamp = dmax*ssamp
Meval = math_utils.bezierM(ssamp, bezierorder)

rleval = torch.matmul(Meval, bc)[0]


fig = plt.figure()
plt.scatter(rl[:,0].numpy(), rl[:,1].numpy(), edgecolor="b", facecolor="none", label="Raceline on subtopic %s" % subtopic)
plt.plot(rleval[0,0].numpy(), rleval[0,1].numpy(), "g*", label="Initial Point")
plt.plot(rleval[:,0].numpy(), rleval[:,1].numpy(), label="best fit bc", c="g")
plt.show()


rldict = {"x": rleval[:,0].numpy().tolist(), "y": rleval[:,1].numpy().tolist(), "z": rleval[:,2].numpy().tolist(), "dist": dsamp[0].numpy().tolist(), "bezier_order" : bezierorder} 

with open(os.path.join(bagdir, "%s.json" % subtopic),"w") as f:
    json.dump(rldict, f, indent=2)