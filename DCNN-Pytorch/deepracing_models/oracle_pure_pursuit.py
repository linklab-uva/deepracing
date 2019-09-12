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
import deepracing
import threading
import numpy.linalg as la
import scipy.integrate as integrate
import socket
import scipy.spatial
import bisect
import traceback
import sys
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
current_motion_data = None
N_error = 100
error_ring_buffer = RingBuffer( N_error, dtype=np.float64 )
throttle_out = 0.0
dt = 1/60.0
speed = 0.0
running = True
velsetpoint = 0.0
sock = None
def velocityControl(pgain, igain):
    global velsetpoint, error_ring_buffer, throttle_out, dt, speed, running
    ierr_max = 50.0
    while running:
        if (current_motion_data is None) or len(current_motion_data.m_carMotionData)==0:
            continue
        vel = deepracing.pose_utils.extractVelocity(current_motion_data,0)
        speed = la.norm(vel)
        perr = velsetpoint - speed
        #print("Current vel error: %f" %(perr))
        error_ring_buffer.append(perr)
        errs = np.array(error_ring_buffer)
        if errs.shape[0]<10:
            continue
        ierr = integrate.simps(errs,dx=dt)
        #print("ierr: %f" %(ierr))
        if ierr>ierr_max:
            ierr=ierr_max
        elif ierr<-ierr_max:
            ierr=-ierr_max
        out = pgain*perr + igain*ierr
        if out<-1.0:
            throttle_out = -1.0
        elif out>1.0:
            throttle_out = 1.0
        else:
            throttle_out = out
        time.sleep(dt)
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
           # print("waiting for data:")
            data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
            current_motion_data.ParseFromString(data)
            #print("received message:", current_motion_data)
    except:
        return
def serve():
    global velsetpoint, current_motion_data, throttle_out, running, speed
    parser = argparse.ArgumentParser(description='Image server.')
    parser.add_argument('address', type=str)
    parser.add_argument('port', type=int)
    parser.add_argument('trackfile', type=str)
    parser.add_argument('--max_velocity', type=float, default=175.0, required=False)
    parser.add_argument('--lookahead_time', type=float, default=3.0, required=False)
    parser.add_argument('--lookahead_gain', type=float, default=0.3, required=False)
    parser.add_argument('--velocity_lookahead_gain', type=float, default=1.5, required=False)
    parser.add_argument('--pgain', type=float, default=0.5, required=False)
    parser.add_argument('--igain', type=float, default=0.5, required=False)
    parser.add_argument('--vmax', type=float, default=175.0, required=False)
    args = parser.parse_args()
    address = args.address
    port = args.port
    trackfile = args.trackfile
    lookahead_time = args.lookahead_time
    data_thread = threading.Thread(target=listenForMotionPackets, args=(address, port))
    data_thread.start()
    vel_control_thread = threading.Thread(target=velocityControl, args=(args.pgain, args.igain))
    vel_control_thread.start()
    # inp = input("Enter anything to continue\n")
    # time.sleep(2.0)
    # vel_control_thread = threading.Thread(target=velocityControl, args=(1.0, 1E-5))
    # vel_control_thread.start()
    # vel_control_thread = threading.Thread(target=velocityControl, args=(1.0, 1E-5))
    # vel_control_thread.start()
    
    controller = py_f1_interface.F1Interface(1)
    controller.setControl(0.0,0.0,0.0)
    L_ = 3.629
    lookahead_gain = args.lookahead_gain
    lookahead_gain_vel = args.velocity_lookahead_gain
    vmax = args.vmax/2.237
    velsetpoint = vmax
    t, x, xdot = deepracing.loadArmaFile(trackfile)
    kdtree = scipy.spatial.KDTree(x)
    print(t.shape)
    print(x.shape)
    print(xdot.shape)
    print(x)
    #t_centered = (t-t[0]).copy()
    xaugmented = np.concatenate((x.copy(),np.ones((x.shape[0],1))), axis=1).transpose()
    try:
        smin = .075
        while running:
            if (current_motion_data is None) or (len(current_motion_data.m_carMotionData)==0):
                continue
            motion_data = current_motion_data.m_carMotionData[0]
            current_pos, current_quat = deepracing.pose_utils.extractPose(current_motion_data)
            current_transform = deepracing.pose_utils.toHomogenousTransform(current_pos, current_quat)
            current_transform_inv = la.inv( current_transform )
            x_local_augmented = np.matmul(current_transform_inv,xaugmented).transpose()
            x_local = x_local_augmented[:,0:3]
            #pquery = current_pos
            #pquery = pquery-(L_/2)*forward
            #nearest_distances, nearest_indices = kdtree.query([pquery])
            #deltas = x-pquery
            distances = la.norm(x_local, axis=1)
            closest_index = np.argmin(distances)
            closest_point = x_local[closest_index]
            closest_t = t[closest_index]
            x_local_forward = x_local[closest_index:]
            # i_splinestart = closest_index
            # i_splineend = bisect.bisect_left(t,closest_t+lookahead_time)
            # tfit = t[i_splinestart:i_splineend]
            # if len(tfit)==0:
            #     continue
            # sfit = (tfit-tfit[0])/(tfit[-1]-tfit[0])
            # xfit = x[i_splinestart:i_splineend]
            # s = max(smin, lookahead_gain*(speed/vmax))
            # if(s>1.0):
            #     s=1.0
            #print(sfit)
           # print(s)
           # xspline = scipy.interpolate.interp1d(sfit,xfit, axis=0, kind='linear')
            lookahead_distance = lookahead_gain*speed
            distances_forward = la.norm(x_local_forward, axis=1)
            lookahead_index = np.argmin(np.abs(distances_forward-lookahead_distance))
            lookaheadVector = x_local_forward[lookahead_index]
            
            lookaheadVector[1]=0.0
            lookaheadVector[2]+=L_/2
            D = la.norm(lookaheadVector)
            lookaheadVector = lookaheadVector/D
            # print()
            # print(pquery)
            # print(lookaheadVector)
            # print()
            # Dvel = la.norm(looakhead_point_vel)
            # lookaheadVectorVel = looakhead_point_vel/Dvel
            alpha = np.abs(np.arccos(np.dot(lookaheadVector,np.array((0,0,1)))))
            if (lookaheadVector[0] < 0):
                alpha *= -1.0
                #alphaVelocity *= -1.0
            physical_angle = np.arctan((2 * L_*np.sin(alpha)) / D)
           # print("Alpha: %f" %(alpha))
            #print("Physical wheel angle desired: %f" %(physical_angle))
            delta = 0.0
            if (physical_angle > 0) :
                delta = 3.79616039*physical_angle# + 0.01004506
            else:
                delta = 3.34446413*physical_angle# + 0.01094534
            #print(delta)
            if throttle_out>0.0:
                controller.setControl(delta,throttle_out,0.0)
            else:
                controller.setControl(delta,0.0,-throttle_out)
            x_vel = np.ones(3)
            
            lookahead_distance_vel = lookahead_gain_vel*speed
            lookahead_index_vel = np.argmin(np.abs(distances_forward-lookahead_distance_vel))
            lookaheadVectorVel = x_local_forward[lookahead_index_vel]
            lookaheadVectorVel = lookaheadVector
            alphavel = np.abs(np.arccos(np.dot(lookaheadVectorVel,np.array((0.0,0.0,1.0)))))
            
            velsetpoint = max(vmax*((1.0-(alphavel/1.57))**7), 25)
            if lookaheadVector[0]<0.0:
                alpha *= -1.0
            time.sleep(0.015)
            
    except KeyboardInterrupt as e:
     #   global running, sock
        controller.setControl(0.0,0.0,0.0)
        running = False
        sock.close()
        time.sleep(0.25)
        exit(0)        
    except Exception as e:
    #    global running, sock
        print(e)
        traceback.print_exc(file=sys.stdout)
        controller.setControl(0.0,0.0,0.0)
        running = False
        sock.close()
        time.sleep(0.25)
        exit(0)
        
  
if __name__ == '__main__':
    logging.basicConfig()
    serve()