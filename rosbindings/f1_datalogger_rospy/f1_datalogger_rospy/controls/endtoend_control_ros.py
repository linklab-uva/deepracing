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
from geometry_msgs.msg import Vector3Stamped, Vector3, PointStamped, Point, PoseStamped, Pose
import rclpy
from rclpy import Parameter
from rclpy.node import Node
import deepracing_models.nn_models.Models as M

class AdmiralNetPurePursuitControllerROS(PPC):
    def __init__(self, trackfile=None, forward_indices = 60,\
         lookahead_gain = 0.4, L = 3.617, pgain=0.5, igain=0.0125, dgain=0.0125, plot=True, gpu=1, deltaT = 1.415):
        super(AdmiralNetPurePursuitControllerROS, self).__init__(lookahead_gain = lookahead_gain, L = L ,\
                                                    pgain=pgain, igain=igain, dgain=dgain)
        
        trackfile = self.get_parameter_or("trackfile", trackfile)
        if (trackfile is not None) and ( not trackfile.type_==Parameter.Type.NOT_SET  ):
            t, x, xdot = deepracing.loadArmaFile(trackfile)
            self.xgt = np.vstack((x.copy().transpose(),np.ones(x.shape[0])))
            self.xdotgt = xdot.copy().transpose()
            self.tgt = t.copy()    
        model_file = self.get_parameter("model_file")
        if (model_file.type_==Parameter.Type.NOT_SET):
            raise ValueError("The parameter \"model_file\" must be set for this rosnode")
        config_file = os.path.join(os.path.dirname(model_file),"config.yaml")
        with open(config_file,'r') as f:
            config = yaml.load(f, Loader = yaml.SafeLoader)
        input_channels = config["input_channels"]
        context_length = config["context_length"]
        bezier_order = config.get("bezier_order",None)
        sequence_length = config.get("sequence_length",None)

        gpu = self.get_parameter_or("gpu",gpu)
        self.gpu = gpu
        L = self.get_parameter_or("wheelbase",L)
        self.L = L

        self.pgain = self.get_parameter_or("pgain",pgain)
        self.igain = self.get_parameter_or("igain",igain)
        self.dgain = self.get_parameter_or("dgain",dgain)
        self.lookahead_gain = self.get_parameter_or("lookahead_gain",lookahead_gain)
        self.plot = self.get_parameter_or("plot",plot)
        self.deltaT = self.get_parameter_or("deltaT",deltaT)
        self.forward_indices = self.get_parameter_or("forward_indices",forward_indices)
        
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
        self.trajplot = None
        self.fig = None
        self.ax = None
        self.cropwidth = self.get_parameter_or("cropwidth",1752)
        self.cropheight = self.get_parameter_or("cropheight",465)
        self.image_sub = self.create_subscription(
            Image,
            '/f1_screencaps',
            self.imageCallback,
            10)
    def imageCallback(self, msg : Image):
        rows = msg.height
        cols = msg.width
        if rows<=0 or cols<=0:
            return
        channels = 3
        imnp = np.frombuffer(msg.data.tobytes(),dtype=np.uint8).reshape(rows,cols,channels)
        imnp = imnp[32:]
        imnp = imnp[0:self.cropheight,0:self.cropwidth,:]
        imnp = deepracing.imutils.resizeImage(imnp,(66,200))
        imnpfloat = imnp.astype(np.float64)/255.0
        self.image_buffer.append(imnpfloat.transpose(2,0,1))
    def getTrajectory(self):
        if self.current_motion_data.world_velocity.header.frame_id == "":
            return None, None, None
        imtorch = torch.from_numpy(np.array(self.image_buffer).copy().astype(np.float64))
        if ( not imtorch.shape[0] == self.net.context_length ):
            return None, None, None
        inputtorch = imtorch
        if isinstance(self.net,  M.AdmiralNetCurvePredictor):
            bezier_control_points = self.net(inputtorch.unsqueeze(0).cuda(self.gpu)).transpose(1,2)
            evalpoints = torch.matmul(self.bezierM, bezier_control_points)
            x_samp = evalpoints[0].cpu().detach().numpy()
            x_samp[:,0]*=1.125
            _, evalvel = mu.bezierDerivative(bezier_control_points,self.s_torch)
            v_samp = (0.925/self.deltaT)*(evalvel[0].cpu().detach().numpy())
        else:
            evalpoints =  self.net(inputtorch.unsqueeze(0).cuda(self.gpu))
            x_samp = evalpoints[0].cpu().detach().numpy()
            x_samp[:,0]*=1.125
            tsamp = np.linspace(0,self.deltaT,60)
            spline = scipy.interpolate.make_interp_spline(tsamp,x_samp)
            splineder = spline.derivative()
            v_samp = splineder(tsamp)
        x_samp[:,1]+=self.L/2
        #print(x_samp)
        distances_samp = la.norm(x_samp, axis=1)
        if self.plot:
            if self.trajplot is None:
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot()
                self.trajplot, = self.ax.plot(-x_samp[:,0],x_samp[:,1], color='b')
                self.ax.set_xlim(-15,15)
                self.ax.set_ylim(0,125)
                plt.show(block=False)
            else:
                self.trajplot.set_xdata(-x_samp[:,0])
                self.trajplot.set_ydata(x_samp[:,1])
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
        return x_samp, v_samp, distances_samp
        