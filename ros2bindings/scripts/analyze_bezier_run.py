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

import deepracing, deepracing.pose_utils, deepracing_models

import numpy as np
import cv2

import f1_datalogger_rospy.convert
from scipy.spatial.transform import Rotation, RotationSpline
from scipy.interpolate import BSpline, make_interp_spline

import deepracing_models, deepracing_models.math_utils
import bisect
import json
import scipy
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

def extractPosition(vectormsg):
    return np.array( [ msg.x, msg.y, msg.z ] )
def imgKey(msg):
    return rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds
def bezierKey(msg):
    return rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds
def msgKey(msg):
    return msg.udp_packet.header.session_time

parser = argparse.ArgumentParser(description="Look for bad predictions in a run of the bezier curve predictor")
parser.add_argument("bag_dir", type=str,  help="Bag to load")
parser.add_argument("--trackfiledir", help="Path to the directory containing the raceline json files. Default is environment variable F1_TRACK_DIR",  type=str, default=os.environ.get("F1_TRACK_DIR",default=None))
parser.add_argument("--save_all_figures", action="store_true",  help="Save a plot for every bezier curve. Even ones with no track violation.")

args = parser.parse_args()

argdict = dict(vars(args))
save_all_figures = argdict["save_all_figures"]
trackfiledir = argdict["trackfiledir"]
if trackfiledir is None:
    raise ValueError("Must either specify --trackfiledir or set environment variable F1_TRACK_DIR")

bag_dir = argdict["bag_dir"]
if bag_dir[-2:] in {"\\\\","//"}:
    bag_dir = bag_dir[0:-2]
elif bag_dir[-1] in {"\\","/"}:
    bag_dir = bag_dir[0:-1]
topic_types, type_map, reader = f1_datalogger_rospy.open_bagfile(bag_dir)
with open(os.path.join(bag_dir,"metadata.yaml"),"r") as f:
    metadata_dict = yaml.load(f,Loader=yaml.SafeLoader)["rosbag2_bagfile_information"]
topic_count_dict = {entry["topic_metadata"]["name"] : entry["message_count"] for entry in metadata_dict["topics_with_message_count"]}
topic_counts = np.array( list(topic_count_dict.values()) ) 

idx = 0
total_msgs = np.sum( topic_counts )
msg_dict = {key : [] for key in topic_count_dict.keys()}
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
session_packet_msgs = sorted(msg_dict["/session_data"], key=msgKey)
motion_packet_msgs = sorted(msg_dict["/motion_data"], key=msgKey)
image_msgs = sorted(msg_dict["/f1_screencaps/full/compressed"], key=imgKey)
bezier_curve_msgs = sorted(msg_dict["/predicted_path"], key=bezierKey)
image_header_timestamps = [rclpy.time.Time.from_msg(msg.header.stamp) for msg in image_msgs]
image_timestamps = np.array([t.nanoseconds/1E9 for t in image_header_timestamps])
bc_header_timestamps = [rclpy.time.Time.from_msg(msg.header.stamp) for msg in bezier_curve_msgs]
bc_timestamps = np.array([t.nanoseconds/1E9 for t in bc_header_timestamps])


motion_timestamps = np.array([rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds/1E9 for msg in motion_packet_msgs])
motion_packet_msgs : List[PacketMotionData] = [msg.udp_packet for msg in motion_packet_msgs]
player_motion_data : List[CarMotionData] = [msg.car_motion_data[msg.header.player_car_index] for msg in motion_packet_msgs]

t0 = bc_timestamps[0]
bc_timestamps = bc_timestamps  - t0
motion_timestamps = motion_timestamps  - t0
image_timestamps = image_timestamps  - t0

print("Extracted %d bezier curves" % ( len(bezier_curve_msgs), ) )
print("Extracted %d motion packets" % ( len(motion_packet_msgs), ) )
tracknum = session_packet_msgs[0].udp_packet.track_id
trackname = deepracing.trackNames[tracknum]
#trackname = "Australia"




inner_boundary_json = os.path.join(trackfiledir, trackname + "_innerlimit.json")
with open(inner_boundary_json,"r") as f:
    inner_boundary_dict = json.load(f)
inner_boundary  = np.column_stack((inner_boundary_dict["x"], inner_boundary_dict["y"], inner_boundary_dict["z"], np.ones_like(inner_boundary_dict["z"]))).transpose()
inner_boundary_dist = inner_boundary_dict["dist"]
ib_xz = inner_boundary[[0,2],:]#.transpose()
ib_atan2 = np.arctan2(ib_xz[1,:], ib_xz[0,:]) + 2.0*np.pi
ib_atan2[ib_atan2>=2*np.pi]-=2.0*np.pi
ib_clocksort = np.flip(np.argsort(ib_atan2))
#ib_poly = inner_boundary[:,ib_clocksort]
ib_poly = inner_boundary
#inner_boundary = inner_boundary[:,ib_clocksort]
print("Inner boundary shape: " + str(inner_boundary.shape))
inner_boundary_kdtree = KDTree(inner_boundary[0:3].transpose())
#inner_boundary_polygon : Polygon = Polygon(reversed(inner_boundary[[0,2],:].transpose().tolist()))
#inner_boundary_polygon : SPPolygon = SPPolygon([SPPoint(inner_boundary[0,i], inner_boundary[2,i]) for i in reversed(range(inner_boundary.shape[1]))])


outer_boundary_json = os.path.join(trackfiledir, trackname + "_outerlimit.json")
with open(outer_boundary_json,"r") as f:
    outer_boundary_dict = json.load(f)
outer_boundary  = np.column_stack((outer_boundary_dict["x"], outer_boundary_dict["y"], outer_boundary_dict["z"], np.ones_like(outer_boundary_dict["z"]))).transpose()
outer_boundary_dist = outer_boundary_dict["dist"]
#print(outer_boundary)
ob_xz = outer_boundary[[0,2]]#.transpose()
ob_atan2 = np.arctan2(ob_xz[1], ob_xz[0]) + 2.0*np.pi
ob_atan2[ob_atan2>=2*np.pi]-=2.0*np.pi
ob_clocksort = np.flip(np.argsort(ob_atan2))
#ob_poly = outer_boundary[:,ob_clocksort]
ob_poly = outer_boundary
#outer_boundary = outer_boundary[:,ob_clocksort]

print("Outer boundary shape: " + str(outer_boundary.shape))
outer_boundary_kdtree = KDTree(outer_boundary[0:3].transpose())
#outer_boundary_polygon : Polygon = Polygon(reversed(outer_boundary[[0,2],:].transpose().tolist()))
#outer_boundary_polygon : SPPolygon = SPPolygon([SPPoint(outer_boundary[0,i], outer_boundary[2,i]) for i in reversed(range(outer_boundary.shape[1]))])

ib_tuples =  LinearRing([(ib_poly[0,i], ib_poly[2,i]) for i in range(ib_poly.shape[1])])
inner_boundary_polygon : Polygon = Polygon(ib_tuples)
assert(inner_boundary_polygon.is_valid)
print("Inner boundary area: %d" % (inner_boundary_polygon.area, ) )


ob_tuples =  LinearRing([(ob_poly[0,i], ob_poly[2,i]) for i in range(ob_poly.shape[1])])
outer_boundary_polygon : Polygon =  Polygon(ob_tuples)
assert(outer_boundary_polygon.is_valid)
print("Outer boundary area: %d" % (outer_boundary_polygon.area, ) )
# ptest = ShapelyPoint(298.902, 724.09)
# assert(not ptest.within(outer_boundary_polygon))


optimal_raceline_json = os.path.join(trackfiledir, trackname + "_racingline.json")
with open(optimal_raceline_json,"r") as f:
    optimal_raceline_dict = json.load(f)
optimal_raceline  = np.column_stack((optimal_raceline_dict["x"], optimal_raceline_dict["y"], optimal_raceline_dict["z"], np.ones_like(optimal_raceline_dict["z"]))).transpose()
optimal_raceline_kdtree = KDTree(optimal_raceline[0:3].transpose())
optimal_raceline_dist = optimal_raceline_dict["dist"]



fig = plt.figure()

ib_recon = np.array(inner_boundary_polygon.exterior.xy).transpose()
# plt.plot(inner_boundary[0],inner_boundary[2], label="Inner Boundary of Track")
plt.plot(ib_recon[:,0],ib_recon[:,1], label="Inner Boundary of Track")

ob_recon = np.array(outer_boundary_polygon.exterior.xy).transpose()
# plt.plot(outer_boundary[0],outer_boundary[2], label="Outer Boundary of Track")
plt.plot(ob_recon[:,0],ob_recon[:,1], label="Outer Boundary of Track")
plt.legend()
plt.show()


bridge = cv_bridge.CvBridge()
poses = [f1_datalogger_rospy.convert.extractPose(msg) for msg in motion_packet_msgs]

positions = np.array( [ pose[0] for pose in poses ] )
position_spline : BSpline = make_interp_spline(motion_timestamps, positions) 
quats = np.array([ pose[1] for pose in poses ])
rotations = Rotation.from_quat(quats)
rotation_spline : RotationSpline = RotationSpline(motion_timestamps, rotations) 

bezier_curves = np.array([ np.column_stack(  [ msg.control_points_lateral, msg.control_points_forward]  ) for msg in bezier_curve_msgs ])
#bezier_curves = np.flip(bezier_curves, axis=1)
with torch.no_grad():
    bezier_curves_torch = torch.from_numpy(bezier_curves.copy()).double()
    bezier_curves_torch = bezier_curves_torch.cuda(0)
    Atorch = deepracing_models.math_utils.bezierM(torch.linspace(0,1,120).unsqueeze(0).double().cuda(0), bezier_curves_torch.shape[1]-1)

    print(bezier_curves_torch[0])
    print(Atorch.shape)
    bcvals = torch.matmul(Atorch[0], bezier_curves_torch)
print(bcvals.shape)
Nsamp = 90
bag_base = os.path.basename(bag_dir)
bag_parent = os.path.join(bag_dir, os.pardir)
figure_dir = os.path.join(bag_parent, bag_base+"_figures")
if os.path.isdir(figure_dir):
    shutil.rmtree(figure_dir)
time.sleep(1.0)
os.makedirs(figure_dir)

boundary_viz_delta = 20
imwriter = None
print("Analyzing bezier curves")
imwriter = None
try:
    for i in tqdm(range(bcvals.shape[0])):
        t = bc_timestamps[i]
        bezier_curve = bezier_curves_torch[i]
        carposition : np.ndarray = position_spline(t)
        carrotation : Rotation = rotation_spline(t)
        homogenous_transform = deepracing.pose_utils.toHomogenousTransform(carposition, carrotation.as_quat())
        homogenous_transform_inv = np.linalg.inv(homogenous_transform)
        bcvalsnp = bcvals[i].cpu().numpy() 
        bcvalslocal = np.row_stack( [bcvalsnp[:,0], np.zeros_like(bcvalsnp[:,0]), bcvalsnp[:,1], np.ones_like(bcvalsnp[:,0]) ] )#.transpose()
        #print(bcvalsaug.shape)
        bcvalsglobal = np.matmul(homogenous_transform, bcvalslocal)#[0:3].transpose()
        bcvalsglobal_shapely = [ShapelyPoint(bcvalsglobal[0,i], bcvalsglobal[2,i]) for i in range(bcvalsglobal.shape[1])]
    
        
        inside = np.array([p.within(inner_boundary_polygon) for p in bcvalsglobal_shapely])
        ibdiffs = np.array([inner_boundary_polygon.exterior.distance(bcvalsglobal_shapely[i]) for i in range(len(bcvalsglobal_shapely))])
        ibdiffs[inside]*=-1.0
        insideviolation = np.any(inside)
        insidepoints = bcvalslocal[:, inside]

        outside = np.array([ not p.within(outer_boundary_polygon) for p in bcvalsglobal_shapely])
        obdiffs = np.array([outer_boundary_polygon.exterior.distance(bcvalsglobal_shapely[i]) for i in range(len(bcvalsglobal_shapely))])
        obdiffs[~outside]*=-1.0
        outsideviolation =  np.any(outside)
        outsidepoints = bcvalslocal[:, outside]



        image_idx = bisect.bisect(image_timestamps, t)
        imnp = bridge.compressed_imgmsg_to_cv2(image_msgs[image_idx],desired_encoding="rgb8")

        _, innerboundary_idx = inner_boundary_kdtree.query(carposition)
        innerboundary_sample_idx = np.flip(np.arange(innerboundary_idx-int(round(Nsamp/4)), innerboundary_idx+Nsamp, step=1, dtype=np.int32)%inner_boundary.shape[1])
        innerboundary_sample = inner_boundary[:,innerboundary_sample_idx]
        innerboundary_sample_local = np.matmul(homogenous_transform_inv,innerboundary_sample)
        #(ibdiffs, ibdiffidx) = inner_boundary_kdtree.query(bcvalsglobal[0:3].transpose())


        _, outerboundary_idx = outer_boundary_kdtree.query(carposition)
        outerboundary_sample_idx = np.flip(np.arange(outerboundary_idx-int(round(Nsamp/4)), outerboundary_idx+Nsamp, step=1, dtype=np.int32)%outer_boundary.shape[1])
        outerboundary_sample = outer_boundary[:,outerboundary_sample_idx]
        outerboundary_sample_local = np.matmul(homogenous_transform_inv,outerboundary_sample)

        _, optimal_raceline_idx = optimal_raceline_kdtree.query(carposition)
        r_orl = optimal_raceline_dist[optimal_raceline_idx]
        optimal_raceline_upperlim = bisect.bisect_left(optimal_raceline_dist, (r_orl+75.0)%optimal_raceline_dist[-1])
        #optimal_raceline_sample_idx = np.flip(np.arange(optimal_raceline_idx, optimal_raceline_upperlim, step=1, dtype=np.int32)%optimal_raceline.shape[1])
        if optimal_raceline_idx<optimal_raceline_upperlim:
            optimal_raceline_sample_idx = np.flip(np.arange(optimal_raceline_idx, optimal_raceline_upperlim, step=1, dtype=np.int32))#%optimal_raceline.shape[1])
        else:
            optimal_raceline_sample_idx = np.flip(np.hstack([np.arange(optimal_raceline_idx, optimal_raceline.shape[1], step=1, dtype=np.int32), np.arange(0, optimal_raceline_upperlim, step=1, dtype=np.int32)]))
        optimal_raceline_sample = optimal_raceline[:,optimal_raceline_sample_idx]
        optimal_raceline_sample_local = np.matmul(homogenous_transform_inv,optimal_raceline_sample)


        
        #(obdiffs, obdiffidx) = outer_boundary_kdtree.query(bcvalsglobal[0:3].transpose())
        anyviolation = insideviolation or outsideviolation# or np.max(np.abs(ibdiffs))<0.1 or np.max(np.abs(obdiffs))<0.1
        allxs = np.hstack([innerboundary_sample_local[0], outerboundary_sample_local[0], optimal_raceline_sample_local[0], bcvalslocal[0] ])
        largestx = np.max(allxs)
        smallestx = np.min(allxs)
        if save_all_figures or anyviolation:# or True:
            fig, (axim, axplot) = plt.subplots(1, 2)
            if save_all_figures:
                axim.set_title("Image Seen By Network at\nCurve %d" % (i,) )
            else:
                axim.set_title("Image Seen By Network.")
            axim.imshow(imnp)
            axplot.set_xlim(largestx+3.0, smallestx-3.0)
            axplot.set_title("Predictions and Track Boundaries")
            ibplot = axplot.plot(innerboundary_sample_local[0], innerboundary_sample_local[2], label="Inner Boundary", c="blue")
            obplot = axplot.plot(outerboundary_sample_local[0], outerboundary_sample_local[2], label="Outer Boundary", c="orange")
            rlplot = axplot.plot(optimal_raceline_sample_local[0], optimal_raceline_sample_local[2], label="Optimal Raceline", c="black")
            predplot = axplot.plot(bcvalslocal[0], bcvalslocal[2], label="Predicted Path", c="green")
            plots = [ibplot, obplot, rlplot, predplot]
            if insideviolation:
                ivplot=axplot.plot(insidepoints[0], insidepoints[2], 'r*')
            if outsideviolation:
                ovplot=axplot.plot(outsidepoints[0], outsidepoints[2], 'r*')
            axplot.plot([0], [0], 'g*', label='Current Position')
            #plt.legend()#loc='center left', bbox_to_anchor=(1, 0.5))
            fig.legend([p[0] for p in plots],\
                        [str(p[0].get_label()) for p in plots],\
                        loc=[0.2,0.1])#\
                        #loc="lower left")#
            if anyviolation:
                figurepath = os.path.join(figure_dir,("curve_%d" % (i,)).upper())
            else:
                figurepath = os.path.join(figure_dir,"curve_%d" % (i,))
            
            fig.savefig(figurepath+".pdf", format="pdf")
            fig.savefig(figurepath+".png", format="png")
            if save_all_figures:
                figure_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)#, sep='')
                figure_img = figure_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                figure_img = cv2.cvtColor(figure_img,cv2.COLOR_RGB2BGR)
                if imwriter is None:
                    fourccformat = "MJPG"
                    fourcc = cv2.VideoWriter_fourcc(*fourccformat)
                    writershape = (figure_img.shape[1], figure_img.shape[0])
                    imwriter = cv2.VideoWriter(os.path.join(figure_dir,"pathvideo.avi"), fourcc, 2, writershape, True)
                imwriter.write(figure_img)
            # else:
            #     axim.set_title("Image Seen By Network")
            plt.close(fig=fig)
            figure_metadata_path = figurepath+"_details.json"
            figure_metadata_dict = {"inside_violation" : int(insideviolation), "outside_violation": int(outsideviolation)}
            if insideviolation:
                figure_metadata_dict["distance_to_boundary"] = np.max(ibdiffs[inside])
            elif outsideviolation:
                figure_metadata_dict["distance_to_boundary"] = np.max(obdiffs[outside])
            else:
                figure_metadata_dict["distance_to_boundary"] = "none"
            
            
            with open(figure_metadata_path, "w") as f:
                json.dump(figure_metadata_dict, f, indent=2, skipkeys=False)
            # if anyviolation:
            #     print()
            #     print("insideviolation: " + str(insideviolation))
            #     print("outsideviolation: " + str(outsideviolation))
            #     print(bcvalsglobal_shapely)
            #     plt.show()
            # else:
            #     plt.close(fig=fig)

        # plt.savefig(os.path.join(figure_dir,"curve_%d.svg" % (i,)), format="svg")
except KeyboardInterrupt as e:
    print("Okie Dokie")
    #imwriter.release()
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