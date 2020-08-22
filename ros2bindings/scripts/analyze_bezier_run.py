import cv_bridge, rclpy, rclpy.time, rclpy.duration, f1_datalogger_rospy
import argparse
import typing
from typing import List

from tqdm import tqdm as tqdm
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from f1_datalogger_msgs.msg import BezierCurve, TimestampedPacketMotionData, PacketMotionData, CarMotionData, PacketHeader
from geometry_msgs.msg import PointStamped, Point, Vector3Stamped, Vector3
from sensor_msgs.msg import CompressedImage

import torch, torchvision

import deepracing, deepracing_models, deepracing_models.math_utils as mu

import numpy as np
import cv2

import f1_datalogger_rospy.convert
from scipy.spatial.transform import Rotation, RotationSpline
from scipy.interpolate import BSpline, make_interp_spline

import deepracing_models


def extractPosition(vectormsg):
    return np.array( [ msg.x, msg.y, msg.z ] )
def msgKey(msg):
    return rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds

parser = argparse.ArgumentParser(description="Look for bad predictions in a run of the bezier curve predictor")
parser.add_argument("bag_dir", type=str,  help="Bag to load")

args = parser.parse_args()

argdict = dict(vars(args))

bag_dir = argdict["bag_dir"]

bridge = cv_bridge.CvBridge()

topic_types, type_map, reader = f1_datalogger_rospy.open_bagfile(bag_dir)
topic_counts = reader.get_topic_counts()
motion_packet_msgs = []
bezier_curve_msgs = []
image_msgs = []
images_np = []
idx = 0
total_msgs = np.sum( np.array( list(topic_counts.values()) ) )
#{'/f1_screencaps/cropped/compressed': 'sensor_msgs/msg/CompressedImage', '/motion_data': 'f1_datalogger_msgs/msg/TimestampedPacketMotionData', '/predicted_path': 'f1_datalogger_msgs/msg/BezierCurve'}
print("Loading data from bag")
for idx in tqdm(iterable=range(total_msgs)):
   # print("Reading message: %d" % (idx,) )
    (topic, data, t) = reader.read_next()
    msg_type = type_map[topic]
    msg_type_full = get_message(msg_type)
    msg = deserialize_message(data, msg_type_full)
    idx=idx+1
    if topic=="/f1_screencaps/cropped/compressed":
        #print(msg.header)
        img_cv = bridge.compressed_imgmsg_to_cv2(msg,desired_encoding="bgr8")
        images_np.append((img_cv, rclpy.time.Time.from_msg(msg.header.stamp) ))
        image_msgs.append(msg)
    elif topic=="/motion_data":
        md : TimestampedPacketMotionData = msg
        motion_packet_msgs.append(md)
    elif topic=="/predicted_path":
        bc : BezierCurve = msg
        bezier_curve_msgs.append(bc)
print("Extracted %d bezier curves" % ( len(bezier_curve_msgs), ) )
print("Extracted %d motion packets" % ( len(motion_packet_msgs), ) )
print("Extracted %d images" % ( len(images_np), ) )


image_timestamps = np.array([t[1].nanoseconds/1E9 for t in images_np])
image_sort = np.argsort(image_timestamps)
image_timestamps = image_timestamps[image_sort]
images_np = np.array([images_np[ image_sort[i] ][0] for i in range(image_sort.shape[0])])

bezier_curve_msgs = sorted(bezier_curve_msgs, key=msgKey)
bezier_curve_timestamps = np.array([rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds/1E9 for msg in bezier_curve_msgs])

timestamped_packet_msgs = sorted(motion_packet_msgs, key=msgKey)
motion_timestamps = np.array([rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds/1E9 for msg in timestamped_packet_msgs])
motion_packet_msgs : List[PacketMotionData] = [msg.udp_packet for msg in timestamped_packet_msgs]
player_motion_data : List[CarMotionData] = [msg.car_motion_data[msg.header.player_car_index] for msg in motion_packet_msgs]
poses = [f1_datalogger_rospy.convert.extractPose(msg) for msg in motion_packet_msgs]

positions = np.array( [ pose[0] for pose in poses ] )
position_spline : BSpline = make_interp_spline(motion_timestamps, positions) 
quats = np.array([ pose[1] for pose in poses ])
rotations = Rotation.from_quat(quats)
rotation_spline : RotationSpline = RotationSpline(motion_timestamps, rotations) 

bezier_curves = np.array([ np.column_stack(  [msg.control_points_lateral, msg.control_points_forward]  ) for msg in bezier_curve_msgs ])
bezier_curves_torch = torch.from_numpy(bezier_curves.copy()).double()
bezier_curves_torch = bezier_curves_torch.cuda(0)


print(bezier_curves_torch[0])

# rotationmats = np.array( [ Rotation.as_matrix(r) for r in rotations] )
# homogenous_transforms = np.tile(np.eye(4), (len(rotations),1,1) )
# homogenous_transforms[:,0:3,0:3] = rotationmats
# homogenous_transforms[:,0:3,3] = positions


#rotation_matrices = np.array( [    ] for msg in  player_motion_data  )
# windowname = "image"
# cv2.namedWindow(windowname, cv2.WINDOW_AUTOSIZE)
# for i in range(images_np.shape[0]-1):
#     imnp = images_np[i]
#     cv2.imshow(windowname, imnp)
#     cv2.waitKey(int(round(1000.0*(image_timestamps[i+1] - image_timestamps[i]))))
# cv2.imshow(windowname, images_np[-1])
# cv2.waitKey(0)


    
# parser.add_argument("inner_track", type=str,  help="Json file for the inner boundaries of the track.")
# parser.add_argument("outer_track", type=str,  help="Json file for the outer boundaries of the track.")