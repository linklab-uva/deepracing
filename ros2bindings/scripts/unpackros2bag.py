import cv_bridge, rclpy, rclpy.time, rclpy.duration, f1_datalogger_rospy
import argparse
import typing
from typing import List

from tqdm import tqdm as tqdm
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from f1_datalogger_msgs.msg import BezierCurve, PacketMotionData, CarMotionData, PacketHeader
from f1_datalogger_msgs.msg import TimestampedPacketMotionData, TimestampedPacketCarStatusData, TimestampedPacketCarTelemetryData, TimestampedPacketLapData, TimestampedPacketSessionData
from geometry_msgs.msg import PointStamped, Point, Vector3Stamped, Vector3
from sensor_msgs.msg import CompressedImage

import torch, torchvision

import deepracing, deepracing.pose_utils, deepracing_models


import f1_datalogger_rospy.convert
from scipy.spatial.transform import Rotation, RotationSpline
from scipy.interpolate import BSpline, make_interp_spline

import deepracing_models, deepracing_models.math_utils
import bisect
import json
import scipy, scipy.stats
from scipy.spatial.kdtree import KDTree
import matplotlib.pyplot as plt
import os
from sympy import Point as SPPoint, Polygon as SPPolygon, pi
from shapely.geometry import Point as ShapelyPoint, MultiPoint#, Point2d as ShapelyPoint2d
from shapely.geometry.polygon import Polygon
from shapely.geometry import LinearRing
import shutil
import time
import cv2
import yaml
import json
from rosidl_runtime_py import message_to_ordereddict, message_to_yaml, set_message_fields
import numpy as np
import io

def extractPosition(vectormsg):
    return np.array( [ msg.x, msg.y, msg.z ] )
def imgKey(msg):
    return rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds
def msgKey(msg):
    return msg.udp_packet.header.session_time

parser = argparse.ArgumentParser(description="Look for bad predictions in a run of the bezier curve predictor")
parser.add_argument("bag_dir", type=str,  help="Bag to load")
parser.add_argument("--viz", action="store_true",  help="Visualize the images from the bag file while they are being written to disk")

args = parser.parse_args()

argdict = dict(vars(args))

bag_dir = argdict["bag_dir"]
if bag_dir[-2:] in {"\\\\","//"}:
    bag_dir = bag_dir[0:-2]
elif bag_dir[-1] in {"\\","/"}:
    bag_dir = bag_dir[0:-1]
viz = argdict["viz"]

bridge = cv_bridge.CvBridge()

topic_types, type_map, reader = f1_datalogger_rospy.open_bagfile(bag_dir)
with open(os.path.join(bag_dir,"metadata.yaml"),"r") as f:
    metadata_dict = yaml.load(f,Loader=yaml.SafeLoader)["rosbag2_bagfile_information"]
topic_count_dict = {entry["topic_metadata"]["name"] : entry["message_count"] for entry in metadata_dict["topics_with_message_count"]}
topic_counts = np.array( list(topic_count_dict.values()) ) 
motion_packet_msgs = []
session_data_msgs = []
telemetry_data_msgs = []
lap_data_msgs = []
status_data_msgs = []
image_msgs = []
idx = 0
total_msgs = np.sum( topic_counts )
#{'/f1_screencaps/cropped/compressed': 'sensor_msgs/msg/CompressedImage', '/motion_data': 'f1_datalogger_msgs/msg/TimestampedPacketMotionData', '/predicted_path': 'f1_datalogger_msgs/msg/BezierCurve'}
print("Loading data from bag")
msg_dict = {key : [] for key in topic_count_dict.keys()}
for idx in tqdm(iterable=range(total_msgs)):
   # print("Reading message: %d" % (idx,) )
    if(reader.has_next()):
        (topic, data, t) = reader.read_next()
        msg_type = type_map[topic]
        msg_type_full = get_message(msg_type)
        msg = deserialize_message(data, msg_type_full)
        msg_dict[topic].append(msg)
#print(message_to_yaml(motion_packet_msgs[-1]))

# image_timestamps = np.array([t.nanoseconds/1E9 for t in image_ros_timestamps])
# image_sort = np.argsort(image_timestamps)
# image_timestamps = image_timestamps[image_sort]
image_msgs = sorted(msg_dict["/f1_screencaps/cropped/compressed"], key=imgKey)
lap_data_msgs = sorted(msg_dict["/lap_data"], key=msgKey)
motion_data_msgs = sorted(msg_dict["/motion_data"], key=msgKey)
session_data_msgs = sorted(msg_dict["/session_data"], key=msgKey)
status_data_msgs = sorted(msg_dict["/status_data"], key=msgKey)
telemetry_data_msgs = sorted(msg_dict["/telemetry_data"], key=msgKey)
#images_np = np.array([images_np[ image_sort[i] ][0] for i in range(image_sort.shape[0])])
print("Extracted %d motion packets" % ( len(motion_data_msgs), ) )
print("Extracted %d images" % ( len(image_msgs), ) )


#rotation_matrices = np.array( [    ] for msg in  player_motion_data  )
root_dir = os.path.join(os.path.dirname(bag_dir), os.path.basename(bag_dir)+"_unpacked")
image_dir = os.path.join(root_dir,"images")
json_dir = os.path.join(root_dir, "json_data")
lap_data_dir = os.path.join(json_dir,"lap_data")
motion_data_dir = os.path.join(json_dir,"motion_data")
session_data_dir = os.path.join(json_dir,"session_data")
status_data_dir = os.path.join(json_dir,"status_data")
telemetry_data_dir = os.path.join(json_dir,"telemetry_data")

os.makedirs(image_dir,exist_ok=True)
os.makedirs(lap_data_dir,exist_ok=True)
os.makedirs(motion_data_dir,exist_ok=True)
os.makedirs(session_data_dir,exist_ok=True)
os.makedirs(status_data_dir,exist_ok=True)
os.makedirs(telemetry_data_dir,exist_ok=True)

print("Writing lap data to json files")
for (i,msg) in tqdm(enumerate(lap_data_msgs), total=len(lap_data_msgs)):
    with open(os.path.join(lap_data_dir, "lap_data_%d.json" % (i+1,) ), "w") as f:
        json.dump(message_to_ordereddict(msg),f,sort_keys=False, indent=2)

print("Writing motion data to json files")
for (i,msg) in tqdm(enumerate(motion_data_msgs), total=len(motion_data_msgs)):
    with open(os.path.join(motion_data_dir, "motion_data_%d.json" % (i+1,) ), "w") as f:
        json.dump(message_to_ordereddict(msg),f,sort_keys=False, indent=2)

print("Writing session data to json files")
for (i,msg) in tqdm(enumerate(session_data_msgs), total=len(session_data_msgs)):
    with open(os.path.join(session_data_dir, "session_data_%d.json" % (i+1,) ), "w") as f:
        json.dump(message_to_ordereddict(msg),f,sort_keys=False, indent=2)

print("Writing status data to json files")
for (i,msg) in tqdm(enumerate(status_data_msgs), total=len(status_data_msgs)):
    with open(os.path.join(status_data_dir, "status_data_%d.json" % (i+1,) ), "w") as f:
        json.dump(message_to_ordereddict(msg),f,sort_keys=False, indent=2)

print("Writing telemetry data to json files")
for (i,msg) in tqdm(enumerate(telemetry_data_msgs), total=len(telemetry_data_msgs)):
    with open(os.path.join(telemetry_data_dir, "telemetry_data_%d.json" % (i+1,) ), "w") as f:
        json.dump(message_to_ordereddict(msg),f,sort_keys=False, indent=2)


print("Generating linear map from ROS time to session time")
motion_packet_ros_times = np.array([float(imgKey(msg))/1E9 for msg in motion_data_msgs])
t0 = motion_packet_ros_times[0]
motion_packet_ros_times = motion_packet_ros_times - t0
motion_packet_session_times = np.array([msgKey(msg) for msg in motion_data_msgs])

slope, intercept, rval, pvalue, stderr = scipy.stats.linregress(motion_packet_ros_times, motion_packet_session_times)
rsquare = rval**2
print("Slope of linear regression line: %f" % (slope,) )
print("Intercept of linear regression line: %f" % (intercept,) )
print("R^2 of linear regression line: %f" % (rsquare,) )
image_session_times = slope*(np.array([(float(imgKey(msg))/1E9) for msg in image_msgs]).astype(np.float64) - t0 ) + intercept


print("Writing images data to jpg files")
if viz:
    windowname = "image"
    cv2.namedWindow(windowname, cv2.WINDOW_AUTOSIZE)
for (i,msg) in tqdm(enumerate(image_msgs)):
    imrgb = bridge.compressed_imgmsg_to_cv2(msg,desired_encoding="rgb8")
    imbgr = cv2.cvtColor(imrgb,cv2.COLOR_RGB2BGR)
    if viz:
        cv2.imshow(windowname, imbgr)
        cv2.waitKey(1)
   # dt = int((rclpy.time.Time.from_msg(msg2.header.stamp).nanoseconds -  rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds)/1E6)
    file_prefix = "image_%d" % (i+1,)
    cv2.imwrite(os.path.join(image_dir,file_prefix+".jpg"), imbgr)
   # im_session_time = slope*(float(imgKey(msg))/1E9) + intercept
    im_session_time = image_session_times[i]
    imdict = {"ros_timestamp" : message_to_ordereddict(msg.header), "session_time" : im_session_time}
    with open(os.path.join(image_dir,file_prefix+".json"),"w") as f:
        json.dump(imdict,f,sort_keys=False, indent=2)


    
# parser.add_argument("inner_track", type=str,  help="Json file for the inner boundaries of the track.")
# parser.add_argument("outer_track", type=str,  help="Json file for the outer boundaries of the track.")