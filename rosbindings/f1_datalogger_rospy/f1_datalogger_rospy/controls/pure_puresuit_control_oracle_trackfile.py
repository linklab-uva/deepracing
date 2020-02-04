import argparse
import skimage
import skimage.io as io
import os
import time
from concurrent import futures
import logging
import argparse
import lmdb
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
from sensor_msgs.msg import Image, CompressedImage
from f1_datalogger_msgs.msg import PathRaw, ImageWithPath
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
import cv_bridge, cv2, numpy as np
import deepracing.pose_utils
import deepracing.protobuf_utils
from deepracing.protobuf_utils import loadTrackfile
from scipy.spatial import KDTree
class OraclePurePursuitControllerROS(PPC):
    def __init__(self, trackfile, forward_indices : int = 60, lookahead_gain : float = 0.4, L : float= 3.617, pgain: float=0.5, igain : float=0.0125, dgain : float=0.0125, plot : bool =True, gpu : int=0, deltaT : float = 1.415):
        super(AdmiralNetBezierPurePursuitControllerROS, self).__init__(lookahead_gain = lookahead_gain, L = L ,\
                                                    pgain=pgain, igain=igain, dgain=dgain)
        trackfileparam : Parameter = self.get_parameter_or("trackfile", Parameter("trackfile") )
        if trackfileparam.type_==Parameter.Type.NOT_SET:
            raise ValueError("\"trackfile\" parameter not set")
        r  , X  = loadTrackfile(trackfile)
        self.r : np.ndarray = r.copy()
        self.X : np.ndarray = X.copy()
        self.kdtree = KDTree(self.X)
        #self.xdotgt = xdot.copy().transpose()
        self.tgt = t.copy()    
        self.path_publisher = self.create_publisher(ImageWithPath, "/predicted_path", 10)
        self.cvbridge : cv_bridge.CvBridge = cv_bridge.CvBridge()


        L_param : Parameter = self.get_parameter_or("wheelbase",Parameter("wheelbase", value=L))
        print("L_param: " + str(L_param))

        pgain_param : Parameter = self.get_parameter_or("pgain",Parameter("pgain", value=pgain))
        print("pgain_param: " + str(pgain_param))

        igain_param : Parameter = self.get_parameter_or("igain",Parameter("igain", value=igain))
        print("igain_param: " + str(igain_param))

        dgain_param : Parameter = self.get_parameter_or("dgain",Parameter("dgain", value=dgain))
        print("dgain_param: " + str(dgain_param))

        lookahead_gain_param : Parameter = self.get_parameter_or("lookahead_gain",Parameter("lookahead_gain", value=lookahead_gain))
        print("lookahead_gain_param: " + str(lookahead_gain_param))

        plot_param : Parameter = self.get_parameter_or("plot",Parameter("plot", value=plot))
        print("plot_param: " + str(plot_param))

        forward_indices_param : Parameter = self.get_parameter_or("forward_indices",Parameter("forward_indices", value=forward_indices))
        print("forward_indices_param: " + str(forward_indices_param))

        x_scale_factor_param : Parameter = self.get_parameter_or("x_scale_factor",Parameter("x_scale_factor", value=1.0))
        print("xscale_factor_param: " + str(x_scale_factor_param))

        z_offset_param : Parameter = self.get_parameter_or("z_offset",Parameter("z_offset", value=L/2.0))
        print("z_offset_param: " + str(z_offset_param))

        
        velocity_scale_param : Parameter = self.get_parameter_or("velocity_scale_factor",Parameter("velocity_scale_factor", value=1.0))
        print("velocity_scale_param: " + str(velocity_scale_param))
        
        num_sample_points_param : Parameter = self.get_parameter_or("num_sample_points",Parameter("num_sample_points", value=60))
        print("num_sample_points_param: " + str(num_sample_points_param))

        self.pgain : float = pgain_param.get_parameter_value().double_value
        self.igain : float = igain_param.get_parameter_value().double_value
        self.dgain : float = dgain_param.get_parameter_value().double_value
        self.lookahead_gain : float = lookahead_gain_param.get_parameter_value().double_value
        self.L = L_param.get_parameter_value().double_value
        self.z_offset : float = z_offset_param.get_parameter_value().double_value
        self.xscale_factor : float = x_scale_factor_param.get_parameter_value().double_value
        self.plot : bool = plot_param.get_parameter_value().bool_value
        self.velocity_scale_factor : float = velocity_scale_param.get_parameter_value().double_value
        self.forward_indices : int = forward_indices_param.get_parameter_value().integer_value
        
        
        self.image_sub = self.create_subscription( Image, '/f1_screencaps/cropped', self.imageCallback, 10)
    
    
    def imageCallback(self, img_msg : Image):
        if img_msg.height<=0 or img_msg.width<=0:
            return
        imnp = self.cvbridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8") 
        imnpdouble = tf.functional.to_tensor(deepracing.imutils.resizeImage( imnp, (66,200) ) ).double().numpy().copy()
        self.image_buffer.append(imnpdouble)
    def getTrajectory(self):
        if self.current_motion_data.world_velocity.header.frame_id == "":
            return None, None, None
        currentposition, currentquat  = deepracing.protobuf_utils.extractPose(self.current_motion_packet)
        currentpose = np.eye(4)
        currentpose[:,3] = currentposition
        currentpose[0:3,0:3] = currentquat.as_dcm()
        d,startindex = self.kdtree.query(currentposition)
        endindex = (startindex + self.forward_indices+1) % (self.X.shape[0])
        if endindex<startindex:
            a = self.X[ startindex : , : ]
            b = self.X[ 0 : endindex , : ]
            segment = np.hstack((a,b))
        else:
            segment = self.X[ startindex : endindex , : ]
        segmentaugmented = np.vstack( ( segment.transpose(), np.ones(self.X.shape[0]) ) )
        x_samp = np.matmul( la.inv(currentpose), segmentaugmented )[ [0,2] , : ].transpose()
        x_samp[:,1]-=self.z_offset
        t_samp = np.linspace(0,1,self.forward_indices)
        spline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline( t_samp, x_samp, k = 3 )
        splinederiv = spline.derivative()
        #print(x_samp)
        distances_samp = la.norm(x_samp, axis=1)
        vectors = splinederiv(t_samp)
        norms = la.norm(vectors, axis=1)
        tangentvectors = vectors/norms[:,None]
        angles = np.arctan2( tangentvectors[:,0], tangentvectors[:,1] )
        angles_scaled = np.abs(angles/(np.pi/2))
        #scale_factors = np.pi/2 - 
        v_samp = 65.0*tangentvectors
       
        return x_samp, v_samp, distances_samp
        