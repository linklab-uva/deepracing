import DeepF1_RPC_pb2_grpc
import DeepF1_RPC_pb2
import Image_pb2
import ChannelOrder_pb2
import PacketMotionData_pb2
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
from nn_models.Models import AdmiralNetPosePredictor 
import yaml
import torch
import torchvision
import torchvision.transforms as tf
import deepracing.imutils
import scipy
import scipy.interpolate
import py_f1_interface
import deepracing.pose_utils
import threading
import numpy.linalg as la
import scipy.integrate as integrate
import socket
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
current_motion_data = None
error_ring_buffer = RingBuffer( 10, dtype=np.float64 )
throttle_out = 0.0
dt = 1/60.0
t = dt*np.linspace(0,9,10)
speed = 0.0
running = True
velsetpoint = 0.0
sock = None
def velocityControl(pgain, igain):
    global error_ring_buffer, throttle_out, dt, t, speed, running
    while running:
        if current_motion_data is None:
            continue
        vel = deepracing.pose_utils.extractVelocity(current_motion_data,0)
        speed = la.norm(vel)
        perr = velsetpoint - speed
        error_ring_buffer.append(perr)
        errs = np.array(error_ring_buffer)
        if errs.shape[0]<10:
            continue
        ierr = integrate.simps(t,errs)
        if ierr>10.0:
            ierr=10.0
        elif ierr<-10.0:
            ierr=-10.0
        out = pgain*perr# + igain*ierr
        if out<-1.0:
            throttle_out = -1.0
        elif out>1.0:
            throttle_out = 1.0
        else:
            throttle_out = out
        time.sleep(1.5*dt)
    #return 'Done'
def listenForMotionPackets(address, port):
    global running, sock, current_motion_data
    current_motion_data = PacketMotionData_pb2.PacketMotionData()
    try:
        UDP_IP = address
        UDP_PORT = port
        sock = socket.socket(socket.AF_INET, # Internet
                            socket.SOCK_DGRAM) # UDP
        sock.bind((UDP_IP, UDP_PORT))
        while running:
            print("waiting for data:")
            data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
            current_motion_data.ParseFromString(data)
            print("received message:", current_motion_data)
    except:
        return
def serve():
    global velsetpoint 
    parser = argparse.ArgumentParser(description='Image server.')
    parser.add_argument('address', type=str)
    parser.add_argument('port', type=int)
    parser.add_argument('trackfile', type=str)
    parser.add_argument('--max_velocity', type=float, default=175.0, required=False)
    parser.add_argument('--lookahead_time', type=float, default=3.0, required=False)
    parser.add_argument('--lookahead_gain', type=float, default=0.5, required=False)
    args = parser.parse_args()
    address = args.address
    port = args.port
    trackfile = args.trackfile
    data_thread = threading.Thread(target=listenForMotionPackets, args=(address, port))
    data_thread.start()
    # inp = input("Enter anything to continue\n")
    # time.sleep(2.0)
    # vel_control_thread = threading.Thread(target=velocityControl, args=(1.0, 1E-5))
    # vel_control_thread.start()
    # vel_control_thread = threading.Thread(target=velocityControl, args=(1.0, 1E-5))
    # vel_control_thread.start()
    
    controller = py_f1_interface.F1Interface(1)
    controller.setControl(0.0,0.0,0.0)
    L_ = 3.629
    lookahead_gain = 0.3
    lookahead_gain_vel = 1.25
    vmax = 175.0/2.237
    velsetpoint = vmax
    try:
        smin = .15
        while True:
            global throttle_out
            # D = la.norm(looakhead_point)
            # Dvel = la.norm(looakhead_point_vel)
            # lookaheadVector = looakhead_point/D
            # lookaheadVectorVel = looakhead_point_vel/Dvel
            # alpha = np.abs(np.arccos(np.dot(lookaheadVector,np.array((0.0,0.0,1.0)))))
            # alphavel = np.abs(np.arccos(np.dot(lookaheadVectorVel,np.array((0.0,0.0,1.0)))))
            # velsetpoint = max(vmax*((1.0-(alphavel/1.57))**1), 25)
            # if lookaheadVector[0]<0.0:
            #     alpha *= -1.0
            time.sleep(0.01)
            
    except KeyboardInterrupt as e:
        global running, sock
        controller.setControl(0.0,0.0,0.0)
        running = False
        sock.close()
        #time.sleep(1)
        exit(0)
  
if __name__ == '__main__':
    logging.basicConfig()
    serve()