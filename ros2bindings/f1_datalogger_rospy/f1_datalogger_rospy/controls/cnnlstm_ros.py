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
from scipy.signal import butter, lfilter
from scipy.signal import freqs, bilinear
from numpy_ringbuffer import RingBuffer as RB
import time
import timeit

class CNNLSTMROS(Node):
    def __init__(self):
        super(CNNLSTMROS,self).__init__('cnnlstm_control', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
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
        context_length = config["context_length"]
        sequence_length = config["sequence_length"]
        output_dimension = config["output_dimension"]
        hidden_dimension = config["hidden_dimension"]


        gpu_param : Parameter = self.get_parameter_or("gpu",Parameter("gpu", value=0))
        print("gpu_param: " + str(gpu_param))

        use_compressed_images_param : Parameter = self.get_parameter_or("use_compressed_images",Parameter("use_compressed_images", value=False))
        print("use_compressed_images_param: " + str(use_compressed_images_param))
        self.gpu = gpu_param.get_parameter_value().integer_value
        self.net : NN.Module = M.CNNLSTM(input_channels=input_channels,context_length=context_length, output_dimension=output_dimension, sequence_length=sequence_length, hidden_dimension=hidden_dimension) 
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
        capacity=10
        self.steer_buffer = RB(capacity)
        self.accel_buffer = RB(capacity)
        cutoff_freq = 12.5 # 1 hz filter
        b,a = butter(3,cutoff_freq,analog=True)
        fs = 90.888099
        self.dt = 1/fs
        z, p = bilinear(b,a,fs=fs)
        self.z = z
        self.p = p


        if use_compressed_images_param.get_parameter_value().bool_value:
            self.image_sub = self.create_subscription( CompressedImage, '/f1_screencaps/cropped/compressed', self.compressedImageCallback, 1)
        else:
            self.image_sub = self.create_subscription( Image, '/f1_screencaps/cropped', self.imageCallback, 1)
        self.control_thread = threading.Thread(target=self.controlLoop)
        self.image_buffer = RB(self.net.context_length,dtype=(float,(3,66,200)))
        self.running=False
        self.timerpub = self.create_publisher(Float64, "/dt", 1)

    def start(self):
        self.running=True
        self.control_thread.start()
    def stop(self):
        self.running=False
    def controlLoop(self):
        while self.running:
            t1 = timeit.default_timer()
            imnp = np.array(self.image_buffer).astype(np.float64).copy()
            imtorch = torch.from_numpy(imnp.copy())
            #print(imtorch.shape)
            if ( not imtorch.shape[0] == self.net.context_length ):
                continue
            controlout = self.net(imtorch.unsqueeze(0).cuda(self.gpu))
            steering = controlout[0,0,0].item()
            differential = controlout[0,0,1].item()
            self.steer_buffer.append(steering)
            self.accel_buffer.append(differential)
            if not (self.steer_buffer.is_full and self.accel_buffer.is_full):
                continue
            steering_filtered = lfilter(self.z,self.p,np.array(self.steer_buffer))
            accel_filtered = np.array(self.accel_buffer)
            #accel_filtered = lfilter(self.z,self.p,np.array(self.accel_buffer))
            steering = 1.5*steering_filtered[-1]
            differential = 10.0*accel_filtered[-1]
            if differential>0:
                self.controller.setControl(-steering, differential, 0.0)
            else:
                self.controller.setControl(-steering, 0.0, -differential)
            t2 = timeit.default_timer()
            dt = t2-t1
            self.timerpub.publish(Float64(data=dt))
            
    def compressedImageCallback(self, img_msg : CompressedImage):
       # print("Got a compressed image")
        try:
            imnp = self.cvbridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding="rgb8") 
        except Exception as e:
            print(e)
            return
       # print(imnp.shape)
        if imnp.shape[0]<=0 or imnp.shape[0]<=0:
            return
        imnpdouble = tf.functional.to_tensor(deepracing.imutils.resizeImage( imnp, (66,200) ) ).double().numpy().copy()
        self.image_buffer.append(imnpdouble)
    def imageCallback(self, img_msg : Image):
        print("Got an image")
        if img_msg.height<=0 or img_msg.width<=0:
            return
        try:
            imnp = self.cvbridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding="rgb8") 
        except:
            print(e)
            return
        imnpdouble = tf.functional.to_tensor(deepracing.imutils.resizeImage( imnp, (66,200) ) ).double().numpy().copy()
        self.image_buffer.append(imnpdouble)





