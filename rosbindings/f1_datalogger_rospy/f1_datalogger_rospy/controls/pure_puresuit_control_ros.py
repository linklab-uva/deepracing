import cv2
import numpy as np
import argparse
import os
import time
import logging
import cv2
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
import deepracing_models.math_utils as mu
import torch
import torch.nn as NN
import torch.utils.data as data_utils
import deepracing_models.nn_models.Models
import matplotlib.pyplot as plt
from f1_datalogger_msgs.msg import TimestampedPacketMotionData, PacketMotionData, CarMotionData
from geometry_msgs.msg import Vector3Stamped, Vector3
from std_msgs.msg import Float64
import rclpy
from rclpy.node import Node
class PurePursuitControllerROS(Node):
    def __init__(self, lookahead_gain = 0.35, L = 3.617,\
        pgain=0.5, igain=0.0125, dgain=0.0125, tau = 0.0):
        super(PurePursuitControllerROS,self).__init__('pure_pursuit_control', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.packet_queue = queue.Queue()
        self.running = True
        self.current_motion_data : CarMotionData  = CarMotionData()
        self.setpoint_publisher = self.create_publisher(Float64, "vel_setpoint", 10)
        self.sock = None
        self.tau = tau
        self.velsetpoint = 0.0
        self.lookahead_gain = lookahead_gain
        self.current_speed = 0.0
        self.throttle_out = 0.0
        self.controller = py_f1_interface.F1Interface(1)
        self.controller.setControl(0.0,0.0,0.0)
        self.L = L
        self.pgain = pgain
        self.igain = igain
        self.dgain = dgain
        self.prev_err = None
        self.integral = 0.0
       # rclpy.spin()
        self.motion_data_sub = self.create_subscription(
            TimestampedPacketMotionData,
            '/motion_data',
            self.velocityControl,
            10)
        self.control_thread = threading.Thread(target=self.lateralControl)
    def start(self):
        self.control_thread.start()
    def stop(self):
        self.running = False
        time.sleep(0.5)
    def velocityControl(self, msg : TimestampedPacketMotionData):
        #print("got some motion data")
        ierr_max = 50.0
        prev_err = 0.0
        dt=0.01
        pgain = self.pgain
        igain = self.igain
        dgain = self.dgain
        packet : PacketMotionData = msg.udp_packet
        motion_data_vec : list = packet.car_motion_data
        if len(motion_data_vec)==0:
            return
        self.current_motion_data : CarMotionData = motion_data_vec[0]
        velrosstamped : Vector3Stamped = self.current_motion_data.world_velocity
        if (velrosstamped.header.frame_id == ""):
            return
        velros : Vector3 = velrosstamped.vector
        vel = np.array( (velros.x, velros.y, velros.z), dtype=np.float64)
        speed = la.norm(vel)
        self.current_speed = speed
        err = self.velsetpoint - speed
        if self.prev_err is None:
            self.prev_err = err
            return
        self.integral += err*dt
        deriv = (err-self.prev_err)/dt
        out = pgain*err + igain*self.integral + dgain*deriv
        if out<-1.0:
            self.throttle_out = -1.0
        elif out>1.0:
            self.throttle_out = 1.0
        else:
            self.throttle_out = out
        self.prev_err = err
    def getTrajectory(self):
        return None, None, None
    def lateralControl(self):
        while self.running:
            #time.sleep(self.tau)
            lookahead_positions, v_local_forward, distances_forward_ = self.getTrajectory()
            if lookahead_positions is None:
                continue
            if distances_forward_ is None:
                distances_forward = la.norm(lookahead_positions, axis=1)
            else:
                distances_forward = distances_forward_
            self.velsetpoint = la.norm(v_local_forward[int(round(self.forward_indices/4))])
            self.setpoint_publisher.publish(Float64(data=3.6*self.velsetpoint))
            lookahead_distance = self.lookahead_gain*self.current_speed
            lookahead_index = np.argmin(np.abs(distances_forward-lookahead_distance))
            lookaheadVector = lookahead_positions[lookahead_index]
            D = la.norm(lookaheadVector)
            lookaheadDirection = lookaheadVector/D
            alpha = np.arctan2(lookaheadDirection[0],lookaheadDirection[1])
            physical_angle = np.arctan((2 * self.L*np.sin(alpha)) / D)
            if (physical_angle > 0) :
                delta = 3.79616039*physical_angle# + 0.01004506
            else:
                delta = 3.34446413*physical_angle# + 0.01094534
            #delta = 0.0
            if self.throttle_out>0.0:
                self.controller.setControl(delta,self.throttle_out,0.0)
            else:
                self.controller.setControl(delta,0.0,-self.throttle_out)
            #print(delta)
