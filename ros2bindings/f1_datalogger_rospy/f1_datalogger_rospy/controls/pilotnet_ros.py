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
import timeit
class PilotNetROS(Node):
    def __init__(self):
        super(PilotNetROS,self).__init__('pilotnet_control', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.controller = py_f1_interface.F1Interface(1)
        self.controller.setControl(0.0,0.0,0.0)
        model_file_param = self.get_parameter("model_file")
        if (model_file_param.type_==Parameter.Type.NOT_SET):
            raise ValueError("The parameter \"model_file\" must be set for this rosnode")
        model_file = model_file_param.get_parameter_value().string_value
        print("Using model file : " + str(model_file))
        config_file = os.path.join(os.path.dirname(model_file),"config.yaml")
        with open(config_file,'r') as f:
            config = yaml.load(f, Loader = yaml.SafeLoader)
        input_channels = config["input_channels"]


        gpu_param : Parameter = self.get_parameter_or("gpu",Parameter("gpu", value=0))
        print("gpu_param: " + str(gpu_param))

        use_compressed_images_param : Parameter = self.get_parameter_or("use_compressed_images",Parameter("use_compressed_images", value=False))
        print("use_compressed_images_param: " + str(use_compressed_images_param))
        self.gpu = gpu_param.get_parameter_value().integer_value
        self.net : NN.Module = M.PilotNet(input_channels=input_channels, output_dim=2) 
        self.net.double()
        self.get_logger().info('Loading model file: %s' % (model_file) )
        self.net.load_state_dict(torch.load(model_file,map_location=torch.device("cpu")))
        self.get_logger().info('Loaded model file: %s' % (model_file) )
        self.get_logger().info('Moving model params to GPU')
        self.net.cuda(self.gpu)
        self.get_logger().info('Moved model params to GPU')
        self.net.eval()    
        self.rosclock = ROSClock()
        self.cvbridge : cv_bridge.CvBridge = cv_bridge.CvBridge()
        self.timerpub = self.create_publisher(Float64, "/dt", 1)


        if use_compressed_images_param.get_parameter_value().bool_value:
            self.image_sub = self.create_subscription( CompressedImage, '/f1_screencaps/cropped/compressed', self.compressedImageCallback, 1)
        else:
            self.image_sub = self.create_subscription( Image, '/f1_screencaps/cropped', self.imageCallback, 1)

    def compressedImageCallback(self, img_msg : CompressedImage):
       # print("Got a compressed image")
        t1 = timeit.default_timer()
        try:
            imnp = self.cvbridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding="rgb8") 
        except Exception as e:
            print(e)
            return
       # print(imnp.shape)
        if imnp.shape[0]<=0 or imnp.shape[0]<=0:
            return
        inputtorch = tf.functional.to_tensor(imnp).unsqueeze(0).double().cuda(self.gpu)
        controls = self.net(inputtorch)
        #print(controls)
        steering = controls[0,0].item()
        differential = controls[0,1].item()
        if differential>0:
            self.controller.setControl(steering, differential,0.0)
        else:
            self.controller.setControl(steering, 0.0, -differential)
        t2 = timeit.default_timer()
        dt = t2 - t1
        self.timerpub.publish(Float64(data=dt))
    def imageCallback(self, img_msg : Image):
        print("Got an image")
        if img_msg.height<=0 or img_msg.width<=0:
            return
        try:
            imnp = self.cvbridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding="rgb8") 
        except:
            print(e)
            return
        
        inputtorch = tf.functional.to_tensor(imnp).unsqueeze(0).double().cuda(self.gpu)
        controls = self.net(inputtorch)[0]
        steering = controls[0].item()
        differential = controls[1].item()
        if differential>0:
            self.controller.setControl(-steering, differential,0.0)
        else:
            self.controller.setControl(-steering, 0.0, -differential)





