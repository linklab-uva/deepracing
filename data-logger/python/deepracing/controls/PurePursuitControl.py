import DeepF1_RPC_pb2_grpc
import DeepF1_RPC_pb2
import Image_pb2
import ChannelOrder_pb2
import PacketMotionData_pb2
import TimestampedPacketMotionData_pb2
import grpc
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
import deepracing.grpc
from numpy_ringbuffer import RingBuffer
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
import bisect
import traceback
import sys
import queue

class PurePursuitController:
    def __init__(self, address="127.0.0.1", port=50052, lookahead_gain = 0.35, L = 3.617,\
        pgain=0.5, igain=0.0125, dgain=0.0125):
        self.packet_queue = queue.Queue()
        self.running = True
        self.current_motion_data = None
        self.sock = None
        self.velsetpoint = 0.0
        self.lookahead_gain = lookahead_gain
        self.current_speed = 0.0
        self.data_thread = threading.Thread(target=self.listenForMotionPackets, args=(address, port, self.packet_queue))
        self.velocity_control_thread = threading.Thread(target=self.velocityControl, args=(pgain, igain, dgain))
        self.lateral_control_thread = threading.Thread(target=self.lateralControl, args=())
        self.throttle_out = 0.0
        self.controller = py_f1_interface.F1Interface(1)
        self.controller.setControl(0.0,0.0,0.0)
        self.L = L
    def velocityControl(self, pgain, igain, dgain):
        global velsetpoint, error_ring_buffer, throttle_out, dt, speed, running
        ierr_max = 50.0
        prev_err = 0.0
        integral = 0.0
        integral_max = 25.0
        dt=0.01
        while self.running:
            if (self.current_motion_data is None) or len(self.current_motion_data.m_carMotionData)==0:
                continue
            vel = deepracing.pose_utils.extractVelocity( self.current_motion_data, car_index = 0 )
            speed = la.norm(vel)
            self.current_speed = speed
            err = self.velsetpoint - speed
            integral += err*dt
            deriv = (err-prev_err)/dt
            #print("ierr: %f" %(ierr))
            if integral>integral_max:
                integral=integral_max
            elif integral<-integral_max:
                integral=-integral_max
            out = pgain*err + igain*integral + dgain*deriv
            if out<-1.0:
                self.throttle_out = -1.0
            elif out>1.0:
                self.throttle_out = 1.0
            else:
                self.throttle_out = out
            time.sleep(dt)
    def listenForMotionPackets(self, address, port, packet_queue : queue.Queue):
        current_packet = TimestampedPacketMotionData_pb2.TimestampedPacketMotionData()
        try:
            if self.sock is not None:
                self.sock.close()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((address, port))
            while self.running:
                data, addr = self.sock.recvfrom(1024) # buffer size is 1024 bytes
                current_packet.ParseFromString(data)
                self.current_motion_data = current_packet.udp_packet
        except Exception as e:
            print(e)
    def lateralControl(self):
        while self.running:
            lookahead_positions, v_local_forward, distances_forward = self.getTrajectory()
            if lookahead_positions is None:
                continue
            self.velsetpoint = 0.8*la.norm(v_local_forward[int(round(self.forward_indices/6))])
            lookahead_distance = self.lookahead_gain*self.current_speed
            lookahead_index = np.argmin(np.abs(distances_forward-lookahead_distance))
            lookaheadVector = lookahead_positions[lookahead_index]
            D = la.norm(lookaheadVector)
            lookaheadDirection = lookaheadVector/D
            alpha = np.arctan2(lookaheadDirection[0],lookaheadDirection[1])
            physical_angle = np.arctan((2 * self.L*np.sin(alpha)) / D)
            delta = 0.0
            if (physical_angle > 0) :
                delta = 3.79616039*physical_angle# + 0.01004506
            else:
                delta = 3.34446413*physical_angle# + 0.01094534
            #print(delta)
            if self.throttle_out>0.0:
                self.controller.setControl(delta,self.throttle_out,0.0)
            else:
                self.controller.setControl(delta,0.0,-self.throttle_out)
            time.sleep(0.025)


    def start(self):
        self.data_thread.start()
        self.velocity_control_thread.start()
        self.lateral_control_thread.start()
    def stop(self):
        self.running = False
        self.sock.close()
    def getTrajectory(self):
        raise NotImplementedError("Must overwrite getTrajectory")

class OraclePurePursuitController(PurePursuitController):
    def __init__(self,trackfile, forward_indices = 60,  address="127.0.0.1", port=50052, lookahead_gain = 0.35, L = 3.617, pgain=0.5, igain=0.0125, dgain=0.0125):
        super(OraclePurePursuitController, self).__init__(address=address, port=port, lookahead_gain = lookahead_gain, L = L ,\
                                                    pgain=pgain, igain=igain, dgain=dgain)
        t, x, xdot = deepracing.loadArmaFile(trackfile)
        self.x = np.vstack((x.copy().transpose(),np.ones(x.shape[0])))
        self.xdot = xdot.copy().transpose()
        self.t = t.copy()
        self.forward_indices = forward_indices
    def getTrajectory(self):
        if self.current_motion_data is None:
            return None, None, None
        motion_data = self.current_motion_data.m_carMotionData[0]
        current_pos, current_quat = deepracing.pose_utils.extractPose(self.current_motion_data)
        deltazmat = np.eye(4)
        deltazmat[2,3] = -self.L/2
        current_transform = np.matmul(deepracing.pose_utils.toHomogenousTransform(current_pos, current_quat), deltazmat)
        current_transform_inv = la.inv(current_transform)
        x_local_augmented = np.matmul(current_transform_inv,self.x)
        x_local = x_local_augmented[[0,2],:].transpose()
        v_local_augmented = np.matmul(current_transform_inv[0:3,0:3],self.xdot)
        v_local = v_local_augmented[[0,2],:].transpose()
        distances = la.norm(x_local, axis=1)
        closest_index = np.argmin(distances)
        forward_idx = np.linspace(closest_index,closest_index+self.forward_indices,self.forward_indices+1).astype(np.int32)%len(distances)
        v_local_forward = v_local[forward_idx]
        x_local_forward = x_local[forward_idx]
        t_forward = self.t[forward_idx]
        deltaT = t_forward[-1]-t_forward[0]
        if deltaT<0.1:
            return x_local_forward, v_local_forward, distances[forward_idx]
        s = (t_forward - t_forward[0])/deltaT
        x_spline = scipy.interpolate.make_interp_spline(s, x_local_forward)
        s_samp = np.linspace(0.0,1.0,96)
        x_samp = x_spline(s_samp)
        t_samp = s_samp*deltaT + t_forward[0]
        distances_samp = la.norm(x_samp, axis=1)
        return x_samp, v_local_forward, distances_samp
        
           
