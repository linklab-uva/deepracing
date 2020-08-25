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
from scipy.spatial.kdtree import KDTree
import cv_bridge, cv2, numpy as np
from scipy.spatial import KDTree
from copy import deepcopy
import json
import torch

class OraclePurePursuitControllerROS(PPC):
    def __init__(self):
        super(OraclePurePursuitControllerROS, self).__init__()
        raceline_file_param : Parameter = self.get_parameter_or("raceline_file", Parameter("raceline_file") )
        if raceline_file_param.type_==Parameter.Type.NOT_SET:
            raise ValueError("\"raceline_file\" parameter not set")
        self.raceline_file = raceline_file_param.get_parameter_value().string_value
        with open(self.raceline_file,"r") as f:
            self.raceline_dictionary = json.load(f)
        self.raceline = torch.cat( [ torch.tensor(self.raceline_dictionary["x"]).unsqueeze(0),\
                                     torch.tensor(self.raceline_dictionary["y"]).unsqueeze(0),\
                                     torch.tensor(self.raceline_dictionary["z"]).unsqueeze(0),\
                                     torch.ones_like(torch.tensor(self.raceline_dictionary["z"])).unsqueeze(0)], dim=0 ).double()

        self.kdtree = KDTree(self.raceline[0:3].numpy().copy().transpose())

        self.raceline_dists = torch.tensor(self.raceline_dictionary["dist"]).double().cuda(0)
        #self.raceline = self.raceline[:,0:-1].cuda(0)
        self.raceline = self.raceline.cuda(0)
        
        self.cvbridge : cv_bridge.CvBridge = cv_bridge.CvBridge()



        plot_param : Parameter = self.get_parameter_or("plot",Parameter("plot", value=False))
        self.plot : bool = plot_param.get_parameter_value().bool_value

        forward_indices_param : Parameter = self.get_parameter_or("forward_indices",Parameter("forward_indices", value=120))
        self.forward_indices : int = forward_indices_param.get_parameter_value().integer_value

        sample_indices_param : Parameter = self.get_parameter_or("sample_indices",Parameter("sample_indices", value=120))
        self.sample_indices : int = sample_indices_param.get_parameter_value().integer_value

        bezier_order_param : Parameter = self.get_parameter_or("bezier_order",Parameter("bezier_order", value=7))
        self.bezier_order : int = bezier_order_param.get_parameter_value().integer_value

        #self.image_sub = self.create_subscription( Image, '/f1_screencaps/cropped', self.imageCallback, 10)
        self.current_pose : PoseStamped = PoseStamped()
        self.current_pose_mat : torch.DoubleTensor = torch.zeros([4,4],dtype=torch.float64)

        self.pose_sub = self.create_subscription( PoseStamped, '/car_pose', self.poseCallback, 1)

        self.s_torch_lstsq = torch.linspace(0,1,self.forward_indices, dtype=torch.float64).unsqueeze(0).cuda(0)
        self.bezierMlstsq = mu.bezierM(self.s_torch_lstsq, self.bezier_order)
        
        self.s_torch_sample = torch.linspace(0,1,self.sample_indices, dtype=torch.float64).unsqueeze(0).cuda(0)
        self.bezierM = mu.bezierM(self.s_torch_sample, self.bezier_order)
        self.bezierMdot = mu.bezierM(self.s_torch_sample, self.bezier_order-1)
        self.bezierMdotdot = mu.bezierM(self.s_torch_sample, self.bezier_order-2)


        

    def poseCallback(self, pose_msg : PoseStamped):
        self.current_pose = pose_msg
        R = torch.from_numpy(Rot.from_quat( [pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w] ).as_matrix().copy()).double()
        v = torch.tensor( [pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z, 1.0] ).double()
        self.current_pose_mat = torch.cat( [torch.cat([R,torch.zeros(3, dtype=torch.float64).unsqueeze(0)], dim=0), v.unsqueeze(1)  ], dim=1 )
        

    def imageCallback(self, img_msg : Image):
        if img_msg.height<=0 or img_msg.width<=0:
            return
        imnp = self.cvbridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8") 
        self.current_image = imnp.copy()
    def getTrajectory(self):
        if not torch.any(self.current_pose_mat.bool()).item():
            return super().getTrajectory()
        current_pose_mat = self.current_pose_mat.clone()
        current_pose_inv = torch.inverse(current_pose_mat)

        (d, Iclosest) = self.kdtree.query(current_pose_mat[0:3,3].numpy())
        
        print("Didn't crash")
        return super().getTrajectory()



        
        #print(x_samp)
        # return x_samp, v_samp, distances_samp, radii
        