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
import deepracing_models.nn_models.Models
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node


class AdmiralNetPurePursuitControllerROS(PPC):
    def __init__(self, model_file, trackfile=None, forward_indices = 60,  address="127.0.0.1", port=50052, lookahead_gain = 0.5, L = 3.617, pgain=0.5, igain=0.0125, dgain=0.0125, plot=True, gpu=1):
        super(AdmiralNetPurePursuitControllerROS, self).__init__(address=address, port=port, lookahead_gain = lookahead_gain, L = L ,\
                                                    pgain=pgain, igain=igain, dgain=dgain, deltaT = 1.415)
        if trackfile is not None:
            t, x, xdot = deepracing.loadArmaFile(trackfile)
            self.x = np.vstack((x.copy().transpose(),np.ones(x.shape[0])))
            self.xdot = xdot.copy().transpose()
            self.t = t.copy()
        self.deltaT = deltaT
        self.gpu = gpu
        self.forward_indices = forward_indices
        self.rosnode = Node("pure_pursuit_control")
        config_file = os.path.join(os.path.dirname(model_file),"config.yaml")
        with open(config_file,'r') as f:
            config = yaml.load(f, Loader = yaml.SafeLoader)
        input_channels = config["input_channels"]
        context_length = config["context_length"]
        bezier_order = config.get("bezier_order",None)
        sequence_length = config.get("sequence_length",None)
        if bezier_order is not None:
            self.net = nn_models.Models.AdmiralNetCurvePredictor(context_length= context_length, input_channels=input_channels, params_per_dimension=bezier_order+1) 
        else:
            hidden_dimension = config["hidden_dimension"]
            self.net = nn_models.Models.AdmiralNetKinematicPredictor(hidden_dim= hidden_dimension, input_channels=input_channels, output_dimension=2, sequence_length=sequence_length, context_length = context_length)
        self.net.load_state_dict(torch.load(model_file,map_location=torch.device("cpu")))
        self.net = self.net.double()
        self.net = self.net.cuda(self.gpu)
        self.image_buffer = RB(self.net.context_length,dtype=(float,(3,66,200)))
        self.optflow_buffer = RB(self.net.context_length,dtype=(float,(2,66,200)))
        if isinstance(self.net,  nn_models.Models.AdmiralNetCurvePredictor):
            self.s_torch = torch.linspace(0,1,60).unsqueeze(0).double().cuda(gpu)
            self.bezierM = mu.bezierM(self.s_torch,self.net.params_per_dimension-1).double().cuda(gpu)
        self.plot = plot
        self.trajplot = None
        self.fig = None
        self.ax = None
    def getTrajectory(self):
        if self.current_motion_data.world_velocity.header.frame_id == "":
            return None, None, None
        if(self.net.input_channels==3):
            imtorch = torch.from_numpy(np.array(self.image_buffer).copy())
            if (not imtorch.shape[0] == self.net.context_length):
                return None, None, None
            inputtorch = imtorch
        else:
            imtorch = torch.from_numpy(np.array(self.image_buffer).copy())
            optflowtorch = torch.from_numpy(np.array(self.optflow_buffer).copy())
            if (not optflowtorch.shape[0] == self.net.context_length) or (not imtorch.shape[0] == self.net.context_length):
                return None, None, None
            inputtorch = torch.cat([imtorch,optflowtorch],dim=1)
        if isinstance(self.net,  nn_models.Models.AdmiralNetCurvePredictor):
            bezier_control_points = self.net(inputtorch.unsqueeze(0).cuda(self.gpu)).transpose(1,2)
            evalpoints = torch.matmul(self.bezierM, bezier_control_points)
            x_samp = evalpoints[0].cpu().detach().numpy()
            x_samp[:,0]*=1.125
            _, evalvel = mu.bezierDerivative(bezier_control_points,self.s_torch)
            v_samp = (0.925)*(1/self.deltaT)*(evalvel[0].cpu().detach().numpy())
        else:
            evalpoints =  self.net(inputtorch.unsqueeze(0).cuda(self.gpu))
            x_samp = evalpoints[0].cpu().detach().numpy()
            x_samp[:,0]*=1.125
            tsamp = np.linspace(0,self.deltaT,60)
            spline = scipy.interpolate.make_interp_spline(tsamp,x_samp)
            splineder = spline.derivative()
            v_samp = splineder(tsamp)
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
        x_samp[:,1]-=self.L/2
        return x_samp, v_samp, distances_samp
        