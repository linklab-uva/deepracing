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
from f1_datalogger_msgs.msg import TimestampedPacketCarStatusData, TimestampedPacketCarTelemetryData, TimestampedPacketMotionData, PacketCarTelemetryData, PacketMotionData, CarMotionData, CarStatusData, CarTelemetryData
from geometry_msgs.msg import Vector3Stamped, Vector3
from std_msgs.msg import Float64
import rclpy
from rclpy.node import Node
from rclpy import Parameter
from copy import deepcopy
class PurePursuitControllerROS(Node):
    def __init__(self, lookahead_gain = 0.35, L = 3.617,\
        pgain=0.5, igain=0.0125, dgain=0.0125, tau = 0.0):
        super(PurePursuitControllerROS,self).__init__('pure_pursuit_control', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.packet_queue = queue.Queue()
        self.running = True
        self.current_motion_packet : PacketMotionData  = PacketMotionData()
        self.current_motion_data : CarMotionData  = CarMotionData()
        self.current_status_data : CarStatusData  = CarStatusData()
        self.current_telemetry_data : CarTelemetryData  = CarTelemetryData()
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
        velocity_lookahead_gain_param : Parameter = self.get_parameter_or("velocity_lookahead_gain", Parameter("velocity_lookahead_gain",value=0.25))
        print("velocity_lookahead_gain_param: " + str(velocity_lookahead_gain_param.get_parameter_value()))
        self.velocity_lookahead_gain : float = velocity_lookahead_gain_param.get_parameter_value().double_value
        
        left_steer_factor_param : Parameter = self.get_parameter_or("left_steer_factor", Parameter("left_steer_factor",value=3.39814))
        print("left_steer_factor_param: " + str(left_steer_factor_param.get_parameter_value()))
        self.left_steer_factor : float = left_steer_factor_param.get_parameter_value().double_value
        
        right_steer_factor_param : Parameter = self.get_parameter_or("right_steer_factor", Parameter("right_steer_factor",value=3.72814))
        print("right_steer_factor_param: " + str(right_steer_factor_param.get_parameter_value()))
        self.right_steer_factor : float = right_steer_factor_param.get_parameter_value().double_value
        
        use_drs_param : Parameter = self.get_parameter_or("use_drs", Parameter("use_drs",value=False))
        self.use_drs : bool = use_drs_param.get_parameter_value().bool_value
        if self.use_drs:
            print("Using DRS")
        else:
            print("Not using DRS")
        
        self.motion_data_sub = self.create_subscription(
            TimestampedPacketMotionData,
            '/motion_data',
            self.velocityControl,
            1)
        self.status_data_sub = self.create_subscription(
            TimestampedPacketCarStatusData,
            '/status_data',
            self.statusUpdate,
            1)
        self.telemetry_data_sub = self.create_subscription(
            TimestampedPacketCarTelemetryData,
            '/telemetry_data',
            self.telemetryUpdate,
            1)
        self.control_thread = threading.Thread(target=self.lateralControl)
    def start(self):
        self.control_thread.start()
    def stop(self):
        self.running = False
        time.sleep(0.5)

    def telemetryUpdate(self, msg : TimestampedPacketCarTelemetryData):
        self.current_telemetry_data = msg.udp_packet.car_telemetry_data[0]

    def statusUpdate(self, msg : TimestampedPacketCarStatusData):
        self.current_status_data = msg.udp_packet.car_status_data[0]

    def velocityControl(self, msg : TimestampedPacketMotionData):
        #print("got some motion data")
        # ierr_max = 50.0
        # prev_err = 0.0
        # dt=0.01
        # pgain = self.pgain
        # igain = self.igain
        # dgain = self.dgain
        packet : PacketMotionData = msg.udp_packet
        self.current_motion_packet = deepcopy(packet)
        motion_data_vec : list = packet.car_motion_data
        if len(motion_data_vec)==0:
            return
        self.current_motion_data = motion_data_vec[0]
        velrosstamped : Vector3Stamped = self.current_motion_data.world_velocity
        if (velrosstamped.header.frame_id == ""):
           return
        velros : Vector3 = velrosstamped.vector
        vel = np.array( (velros.x, velros.y, velros.z), dtype=np.float64)
        speed = la.norm(vel)
        self.current_speed = speed
        # err = self.velsetpoint - speed
        # if self.prev_err is None:
        #     self.prev_err = err
        #     return
        # self.integral += err*dt
        # deriv = (err-self.prev_err)/dt
        # out = pgain*err + igain*self.integral + dgain*deriv
        # if out<-1.0:
        #     self.throttle_out = -1.0
        # elif out>1.0:
        #     self.throttle_out = 1.0
        # else:
        #     self.throttle_out = out
        # self.prev_err = err
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
            forward_vels = v_local_forward.shape[0]
            self.velsetpoint = la.norm(v_local_forward[int(round(self.velocity_lookahead_gain*(forward_vels-1)))])
            self.setpoint_publisher.publish(Float64(data=3.6*self.velsetpoint))
            # velrosstamped : Vector3Stamped = deepcopy(self.current_motion_data.world_velocity)
            # if (velrosstamped.header.frame_id == ""):
            #     return
            # velros : Vector3 = velrosstamped.vector
            # vel = np.array( (velros.x, velros.y, velros.z), dtype=np.float64)
            # speed = la.norm(vel)
            lookahead_distance = self.lookahead_gain*self.current_speed
            lookahead_index = np.argmin(np.abs(distances_forward-lookahead_distance))
            lookaheadVector = lookahead_positions[lookahead_index]
            D = la.norm(lookaheadVector)
            lookaheadDirection = lookaheadVector/D
            alpha = np.arctan2(lookaheadDirection[0],lookaheadDirection[1])
            physical_angle = np.arctan((2 * self.L*np.sin(alpha)) / D)
            if (physical_angle > 0) :
                delta = self.left_steer_factor*physical_angle# + 0.01004506
            else:
                delta = self.right_steer_factor*physical_angle# + 0.01094534
            #delta = 0.0
            if self.velsetpoint>self.current_speed:
                self.controller.setControl(delta,1.0,0.0)
            else:
                self.controller.setControl(delta,0.0,1.0)
            if self.use_drs and self.current_status_data.m_drs_allowed==1 and self.current_telemetry_data.drs==0:
                self.controller.pushDRS()
            #print(delta)
            # if self.throttle_out>0.0:
            #     self.controller.setControl(delta,self.throttle_out,0.0)
            # else:
            #     self.controller.setControl(delta,0.0,-self.throttle_out)
            #print(delta)
