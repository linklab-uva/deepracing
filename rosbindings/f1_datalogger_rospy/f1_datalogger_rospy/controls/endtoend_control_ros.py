import cv2
import numpy as np
import argparse
import skimage
import skimage.io as io
import os
import time
from concurrent import futures
import logging
import argparse
import lmdb
import cv2
import deepracing.backend
from numpy_ringbuffer import RingBuffer as RB
import yaml
import torch
import torchvision
import torchvision.transforms as tf
import deepracing.imutils
import scipy
import scipy.interpolate
import py_f1_interface
import deepracing.pose_utils
import deepracing
import threading
import numpy.linalg as la
import scipy.integrate as integrate
import socket
import scipy.spatial
import queue
from f1_datalogger_rospy.controls.pure_puresuit_control_ros import PurePursuitControllerROS as PPC
import deepracing_models.math_utils as mu
import torch
import torch.nn as NN
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from f1_datalogger_msgs.msg import PathRaw
from geometry_msgs.msg import Vector3Stamped, Vector3, PointStamped, Point, PoseStamped, Pose, Quaternion
from nav_msgs.msg import Path
from std_msgs.msg import Float64, Header
import rclpy
from rclpy import Parameter
from rclpy.node import Node
from rclpy.time import Time
from rclpy.clock import Clock, ROSClock
import deepracing_models.nn_models.Models as M
from scipy.spatial.transform import Rotation as Rot
name_to_dtypes = {
	"rgb8":    (np.uint8,  3),
	"rgba8":   (np.uint8,  4),
	"rgb16":   (np.uint16, 3),
	"rgba16":  (np.uint16, 4),
	"bgr8":    (np.uint8,  3),
	"bgra8":   (np.uint8,  4),
	"bgr16":   (np.uint16, 3),
	"bgra16":  (np.uint16, 4),
	"mono8":   (np.uint8,  1),
	"mono16":  (np.uint16, 1),
	
    # for bayer image (based on cv_bridge.cpp)
	"bayer_rggb8":	(np.uint8,  1),
	"bayer_bggr8":	(np.uint8,  1),
	"bayer_gbrg8":	(np.uint8,  1),
	"bayer_grbg8":	(np.uint8,  1),
	"bayer_rggb16":	(np.uint16, 1),
	"bayer_bggr16":	(np.uint16, 1),
	"bayer_gbrg16":	(np.uint16, 1),
	"bayer_grbg16":	(np.uint16, 1),

    # OpenCV CvMat types
	"8UC1":    (np.uint8,   1),
	"8UC2":    (np.uint8,   2),
	"8UC3":    (np.uint8,   3),
	"8UC4":    (np.uint8,   4),
	"8SC1":    (np.int8,    1),
	"8SC2":    (np.int8,    2),
	"8SC3":    (np.int8,    3),
	"8SC4":    (np.int8,    4),
	"16UC1":   (np.uint16,   1),
	"16UC2":   (np.uint16,   2),
	"16UC3":   (np.uint16,   3),
	"16UC4":   (np.uint16,   4),
	"16SC1":   (np.int16,  1),
	"16SC2":   (np.int16,  2),
	"16SC3":   (np.int16,  3),
	"16SC4":   (np.int16,  4),
	"32SC1":   (np.int32,   1),
	"32SC2":   (np.int32,   2),
	"32SC3":   (np.int32,   3),
	"32SC4":   (np.int32,   4),
	"32FC1":   (np.float32, 1),
	"32FC2":   (np.float32, 2),
	"32FC3":   (np.float32, 3),
	"32FC4":   (np.float32, 4),
	"64FC1":   (np.float64, 1),
	"64FC2":   (np.float64, 2),
	"64FC3":   (np.float64, 3),
	"64FC4":   (np.float64, 4)
}

def image_to_numpy(msg : Image):
	if not msg.encoding in name_to_dtypes:
		raise TypeError('Unrecognized encoding {}'.format(msg.encoding))
	dtype_class, channels = name_to_dtypes[msg.encoding]
	dtype = np.dtype(dtype_class)
	dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
	shape = (msg.height, msg.width, channels)

	data = np.fromstring(msg.data.tostring(), dtype=dtype).reshape(shape)
	data.strides = (
		msg.step,
		dtype.itemsize * channels,
		dtype.itemsize
	)

	if channels == 1:
		data = data[...,0]
	return data
def npTrajectoryToROS(trajectory : np.ndarray, velocities : np.ndarray, frame_id = "map"):
    rtn : Path = Path()
    rtn.header.frame_id = frame_id
    for i in range(trajectory.shape[0]):
        point = trajectory[i]
        forward = np.array((velocities[i,0],0.0,velocities[i,1]), dtype=np.float64)
        up = np.array((0.0,1.0,0.0), dtype=np.float64)
        left = np.cross(up,forward)
        left[2] = 0.0
        left = left / la.norm(left)
        trueup =  np.cross(forward,left)
        trueup = trueup / la.norm(trueup)

        posestamped : PoseStamped = PoseStamped()
        posestamped.header.frame_id = frame_id
        pose : Pose = Pose()
        pose.position.x = point[0]
        pose.position.z = point[1]
       # pose.position.y = 0
       # pose.position.z = 0
        r = Rot.from_dcm(np.vstack((left, trueup, forward)).transpose())
        quat = r.as_quat()
        pose.orientation = Quaternion()
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        posestamped.pose = pose
        rtn.poses.append(posestamped)
    return rtn
class AdmiralNetPurePursuitControllerROS(PPC):
    def __init__(self, trackfile=None, forward_indices : int = 60, lookahead_gain : float = 0.4, L : float= 3.617, pgain: float=0.5, igain : float=0.0125, dgain : float=0.0125, plot : bool =True, gpu : int=0, deltaT : float = 1.415):
        super(AdmiralNetPurePursuitControllerROS, self).__init__(lookahead_gain = lookahead_gain, L = L ,\
                                                    pgain=pgain, igain=igain, dgain=dgain)
        
        trackfile = self.get_parameter_or("trackfile", trackfile)
        if (trackfile is not None) and ( not trackfile.type_==Parameter.Type.NOT_SET  ):
            t, x, xdot = deepracing.loadArmaFile(trackfile)
            self.xgt = np.vstack((x.copy().transpose(),np.ones(x.shape[0])))
            self.xdotgt = xdot.copy().transpose()
            self.tgt = t.copy()    
        self.path_publisher = self.create_publisher(PathRaw, "predicted_path_raw", 10)
        model_file_param = self.get_parameter("model_file")
        if (model_file_param.type_==Parameter.Type.NOT_SET):
            raise ValueError("The parameter \"model_file\" must be set for this rosnode")
        model_file = model_file_param.get_parameter_value().string_value
        print("Using model file : " + str(model_file))
        config_file = os.path.join(os.path.dirname(model_file),"config.yaml")
        with open(config_file,'r') as f:
            config = yaml.load(f, Loader = yaml.SafeLoader)
        input_channels = config["input_channels"]
        context_length = config["context_length"]
        bezier_order = config.get("bezier_order",None)
        sequence_length = config.get("sequence_length",None)
        self.rosclock = ROSClock()
        #self.rosclock._set_ros_time_is_active(True)


        L = self.get_parameter_or("wheelbase",L)
        self.L = L

        pgain_param : Parameter = self.get_parameter_or("pgain",pgain)
        print("pgain_param: " + str(pgain_param))

        igain_param : Parameter = self.get_parameter_or("igain",igain)
        print("igain_param: " + str(igain_param))

        dgain_param : Parameter = self.get_parameter_or("dgain",dgain)
        print("dgain_param: " + str(dgain_param))

        lookahead_gain_param : Parameter = self.get_parameter_or("lookahead_gain",lookahead_gain)
        print("lookahead_gain_param: " + str(lookahead_gain_param))

        plot_param : Parameter = self.get_parameter_or("plot",plot)
        print("plot_param: " + str(plot_param))

        deltaT_param : Parameter = self.get_parameter_or("deltaT",deltaT)
        print("deltaT_param: " + str(deltaT_param))

        forward_indices_param : Parameter = self.get_parameter_or("forward_indices",forward_indices)
        print("forward_indices_param: " + str(forward_indices_param))

        x_scale_factor_param : Parameter = self.get_parameter_or("x_scale_factor",1.0)
        print("xscale_factor_param: " + str(x_scale_factor_param))

        z_offset_param : Parameter = self.get_parameter_or("z_offset",L/2.0)
        print("z_offset_param: " + str(z_offset_param))

        gpu_param = self.get_parameter_or("gpu",gpu)
        print("gpu_param: " + str(gpu_param))

        self.pgain : float = pgain_param
        self.igain : float = igain_param
        self.dgain : float = dgain_param
        if isinstance(lookahead_gain_param, Parameter):
            self.lookahead_gain : float = lookahead_gain_param.get_parameter_value().double_value
        else:
            self.lookahead_gain : float = lookahead_gain_param
        if isinstance(z_offset_param, Parameter):
            self.z_offset : float = z_offset_param.get_parameter_value().double_value
        else:
            self.z_offset : float = L/2.0
        if isinstance(gpu_param, Parameter):
            gpu : int = gpu_param.get_parameter_value().integer_value
        else:
            gpu : int = gpu
        self.gpu = gpu
        if isinstance(x_scale_factor_param, Parameter):
            self.xscale_factor : float = x_scale_factor_param.get_parameter_value().double_value
        else:
            self.xscale_factor : float = 1.0
        if isinstance(plot_param, Parameter):
            self.plot : bool = plot_param.get_parameter_value().bool_value
        else:
            self.plot : bool = plot
        self.deltaT : float = deltaT_param
        self.forward_indices : int = forward_indices_param
        
        if bezier_order is not None:
            self.net : NN.Module = M.AdmiralNetCurvePredictor(context_length= context_length, input_channels=input_channels, params_per_dimension=bezier_order+1) 
        else:
            hidden_dimension = config["hidden_dimension"]
            self.net : NN.Module = M.AdmiralNetKinematicPredictor(hidden_dim= hidden_dimension, input_channels=input_channels, output_dimension=2, sequence_length=sequence_length, context_length = context_length)
        self.net.double()
        self.net.load_state_dict(torch.load(model_file,map_location=torch.device("cpu")))
        self.net.cuda(gpu)
        self.net.eval()
        self.image_buffer = RB(self.net.context_length,dtype=(float,(3,66,200)))
        if isinstance(self.net,  M.AdmiralNetCurvePredictor):
            self.s_torch = torch.linspace(0,1,60).unsqueeze(0).double().cuda(gpu)
            self.bezierM = mu.bezierM(self.s_torch,self.net.params_per_dimension-1).double().cuda(gpu)
            self.bezierMderiv = mu.bezierM(self.s_torch,self.net.params_per_dimension-2).double().cuda(gpu)
        self.trajplot = None
        self.fig = None
        self.ax = None
        cropwidthparam : Parameter = self.get_parameter_or("cropwidth",1758)
        cropheightparam : Parameter = self.get_parameter_or("cropheight",362)
        self.cropheight = cropheightparam.get_parameter_value().integer_value
        self.cropwidth = cropwidthparam.get_parameter_value().integer_value
        self.image_sub = self.create_subscription(
            Image,
            '/f1_screencaps',
            self.imageCallback,
            10)
       # cv2.namedWindow("imrecv", cv2.WINDOW_AUTOSIZE)
    def imageCallback(self, img_msg : Image):
        if img_msg.height<=0 or img_msg.width<=0:
            return
        n_channels = 3
        imnpbgra = image_to_numpy( img_msg )
        imnp = cv2.cvtColor(imnpbgra,cv2.COLOR_BGRA2RGB)
        imnp = imnp[32:]
        if self.cropheight > 0:
            cropheight = self.cropheight
        else:
            cropheight = imnp.shape[0]
        if self.cropwidth > 0:
            cropwidth = self.cropwidth
        else:
            cropwidth = imnp.shape[1]
        imnp = imnp[0:cropheight,0:cropwidth]
        imnp = deepracing.imutils.resizeImage(imnp,(66,200))
        # cv2.imshow("imrecv", cv2.cvtColor(imnp,cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)
        imnpfloat = ((imnp.astype(np.float64))/255.0).transpose(2,0,1)
        self.image_buffer.append(imnpfloat)
    def getTrajectory(self):
        if self.current_motion_data.world_velocity.header.frame_id == "":
            return None, None, None
        imtorch = torch.from_numpy(np.array(self.image_buffer).copy().astype(np.float64))
        if ( not imtorch.shape[0] == self.net.context_length ):
            return None, None, None
        inputtorch = imtorch
        if isinstance(self.net,  M.AdmiralNetCurvePredictor):
            bezier_control_points = self.net(inputtorch.unsqueeze(0).cuda(self.gpu)).transpose(1,2)
            stamp = self.rosclock.now().to_msg()
            evalpoints = torch.matmul(self.bezierM, bezier_control_points)
            x_samp = evalpoints[0].cpu().detach().numpy()
            x_samp[:,0]*=self.xscale_factor
            _, evalvel = mu.bezierDerivative(bezier_control_points, M = self.bezierMderiv)
            v_samp = (1.0/self.deltaT)*(evalvel[0].cpu().detach().numpy())
        else:
            evalpoints =  self.net(inputtorch.unsqueeze(0).cuda(self.gpu))
            stamp = self.rosclock.now().to_msg()
            x_samp = evalpoints[0].cpu().detach().numpy()
            x_samp[:,0]*=self.xscale_factor
            tsamp = np.linspace(0,self.deltaT,60)
            spline = scipy.interpolate.make_interp_spline(tsamp,x_samp)
            splineder = spline.derivative()
            v_samp = splineder(tsamp)
        x_samp[:,1]-=self.z_offset
        #print(x_samp)
        distances_samp = la.norm(x_samp, axis=1)
        if self.plot:
            PathRawMsg : PathRaw = PathRaw(header = Header(frame_id = "car", stamp = stamp), posx = x_samp[:,0], posz = x_samp[:,1], velx = v_samp[:,0], velz = v_samp[:,1]  )
            self.path_publisher.publish(PathRawMsg)
            # if self.trajplot is None:
            #     self.fig = plt.figure()
            #     self.ax = self.fig.add_subplot()
            #     self.trajplot, = self.ax.plot(-x_samp[:,0],x_samp[:,1], color='b')
            #     self.ax.set_xlim(-15,15)
            #     self.ax.set_ylim(0,125)
            #     plt.show(block=False)
            # else:
            #     self.trajplot.set_xdata(-x_samp[:,0])
            #     self.trajplot.set_ydata(x_samp[:,1])
            #     self.fig.canvas.draw()
            #     self.fig.canvas.flush_events()
        return x_samp, v_samp, distances_samp
        