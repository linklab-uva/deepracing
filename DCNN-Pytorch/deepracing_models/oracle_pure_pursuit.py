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
import google.protobuf.json_format
import matplotlib.pyplot as plt
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
def velocityControl(pgain, igain, setpoint_queue : list, actual_queue : list):
    global velsetpoint, error_ring_buffer, throttle_out, dt, speed, running
    ierr_max = 50.0
    while running:
        if (current_motion_data is None) or len(current_motion_data.m_carMotionData)==0:
            continue
        vel = deepracing.pose_utils.extractVelocity(current_motion_data,0)
        speed = la.norm(vel)
        if (setpoint_queue is not None) and (actual_queue is not None):
            setpoint_queue.append(velsetpoint)
            actual_queue.append(speed)
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

def writeMotionPackets(packet_queue : queue.Queue, logdir : str):
    global running
    counter = 1
    while running or (not packet_queue.empty()):
        if(packet_queue.empty()):
            continue
        packet = packet_queue.get(block=False)
        with open(os.path.join(logdir,"packet_"+str(counter)+".json"), "w") as f:
            json_string = google.protobuf.json_format.MessageToJson(packet, including_default_value_fields=True, indent=0)
            #f.write(packet.SerializeToString())
            f.write(json_string)
            counter = counter + 1
def listenForMotionPackets(address, port, packet_queue : queue.Queue):
    global running, sock, current_motion_data
    current_packet = TimestampedPacketMotionData_pb2.TimestampedPacketMotionData()
    prev_packet = TimestampedPacketMotionData_pb2.TimestampedPacketMotionData()
    current_motion_data = PacketMotionData_pb2.PacketMotionData()
    try:
        UDP_IP = address
        UDP_PORT = port
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        while running:
           # print("waiting for data:")
            data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
            current_packet.ParseFromString(data)
            if (prev_packet is not None) and (current_packet.udp_packet.m_header.m_sessionTime==prev_packet.udp_packet.m_header.m_sessionTime):
                continue
            current_motion_data = current_packet.udp_packet
            if packet_queue is not None:
                packet_queue.put(current_packet)
            prev_packet.CopyFrom(current_packet)
            #print("received message:", current_motion_data)
    except:
        return
def serve():
    global velsetpoint, current_motion_data, throttle_out, running, speed
    parser = argparse.ArgumentParser(description='Image server.')
    parser.add_argument('address', type=str)
    parser.add_argument('port', type=int)
    parser.add_argument('trackfile', type=str)
    parser.add_argument('--lookahead_time', type=float, default=1.0, required=False)
    parser.add_argument('--lookahead_gain', type=float, default=0.3, required=False)
    parser.add_argument('--velocity_lookahead_gain', type=float, default=1.5, required=False)
    parser.add_argument('--pgain', type=float, default=0.5, required=False)
    parser.add_argument('--igain', type=float, default=0.5, required=False)
    parser.add_argument('--vmax', type=float, default=175.0, required=False)
    parser.add_argument('--logdir', type=str, default=None, required=False)
    parser.add_argument('--usesplines', action="store_true")
    args = parser.parse_args()
    address = args.address
    port = args.port
    trackfile = args.trackfile
    lookahead_time = args.lookahead_time
    logdir = args.logdir
    packet_queue = None
    if (logdir is not None):
        packet_queue = queue.Queue()
        os.makedirs(logdir,exist_ok=True)
        logging_thread = threading.Thread(target=writeMotionPackets, args=(packet_queue, logdir))
        logging_thread.start()
    
    data_thread = threading.Thread(target=listenForMotionPackets, args=(address, port, packet_queue))
    data_thread.start()
    setpoint_queue = []
    actual_queue  = []
    vel_control_thread = threading.Thread(target=velocityControl, args=(args.pgain, args.igain, setpoint_queue , actual_queue))
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
    print(t.shape)
    print(x.shape)
    print(xdot.shape)
    print(x)
    #t_centered = (t-t[0]).copy()
    xaugmented = np.concatenate((x.copy(),np.ones((x.shape[0],1))), axis=1).transpose()
    deltazmat = np.eye(4)
    deltazmat[2,3] = -L_/2
    try:
        smin = .2
        plt.show()
        ax = plt.gca()
        hlfit, = ax.plot([], [],'r-')
        hlspline, = ax.plot([], [],'b+')
        plt.xlim(-25,25)
        plt.ylim(0,100)
        while running:
            if (current_motion_data is None) or (len(current_motion_data.m_carMotionData)==0):
                continue
            motion_data = current_motion_data.m_carMotionData[0]
            current_pos, current_quat = deepracing.pose_utils.extractPose(current_motion_data)
            current_transform = np.matmul(deepracing.pose_utils.toHomogenousTransform(current_pos, current_quat), deltazmat)
            current_transform_inv = deepracing.pose_utils.inverseTransform(current_transform)
            x_local_augmented = np.matmul(current_transform_inv,xaugmented).transpose()
            x_local = x_local_augmented[:,0:3]
            v_local = np.matmul(current_transform_inv[0:3,0:3],xdot.transpose()).transpose()
           
            distances = la.norm(x_local, axis=1)
            closest_index = np.argmin(distances)
            forward_idx = (np.linspace(closest_index,closest_index+200,201).astype(np.int32))%(len(distances))
            closest_point = x_local[forward_idx[0]]
            v_local_forward = v_local[forward_idx]
            closest_velocity = v_local_forward[0]
            closest_t = t[closest_index]
            x_local_forward = x_local[forward_idx]
            t_forward = t[forward_idx]
            lookahead_distance = lookahead_gain*speed
            distances_forward = la.norm(x_local_forward, axis=1)
            lookahead_index = np.argmin(np.abs(distances_forward-lookahead_distance))

            if args.usesplines:
                #imax = bisect.bisect_left(t_forward,t_forward[0]+lookahead_time)
                tfit = t_forward[:60]
                #print(tfit)
                if len(tfit)==0:
                    continue
                deltaT = (tfit[-1]-tfit[0])
                if deltaT<0.05:
                    continue
                sfit = (tfit-tfit[0])/deltaT
                xfit = x_local_forward[:60,[0,2]]
                vfit = v_local_forward[:60,[0,2]]
                # print(sfit)
                # print(s)
                xspline = scipy.interpolate.make_interp_spline(sfit,xfit) 
                tsamp = np.linspace(0,1,64)
                xsamp =xspline(tsamp)
                # hlfit, = ax.plot(xfit[:,0], xfit[:,1],'r-')
                # hlspline, = ax.plot(xsamp[:,0], xsamp[:,1],'b-')
                hlfit.set_xdata(xfit[:,0])
                hlfit.set_ydata(xfit[:,1])
                hlspline.set_xdata(xsamp[:,0])
                hlspline.set_ydata(xsamp[:,1])
                velspline = scipy.interpolate.make_interp_spline(sfit,vfit)
                speedfactor = lookahead_gain*(speed/vmax)
                s = min(max(smin, lookahead_distance/(speed*deltaT)),1.0)
                #print(speed)
                #print(deltaT)
                if(s>1.0):
                    s=1.0
                lookaheadVector = xspline(s)
                lookaheadVel = xspline(0.1,nu=1)
                #lookaheadVel = velspline(0.1)*deltaT
                lookaheadAccel = xspline(0.1,nu=2)
                velsetpoint = 0.90*la.norm(lookaheadVel)/deltaT
                #velsetpoint = 0.8*la.norm(v_local_forward[lookahead_index])
            else:
                lookaheadVector = x_local_forward[lookahead_index,[0,2]]
                velsetpoint = la.norm(closest_velocity)
            
            #print(velsetpoint)
            D = la.norm(lookaheadVector)
            lookaheadVector = lookaheadVector/D
            # print()
            # print(pquery)
            # print(lookaheadVector)
            # print()
            # Dvel = la.norm(looakhead_point_vel)
            # lookaheadVectorVel = looakhead_point_vel/Dvel
            #alpha = np.abs(np.arccos(np.dot(lookaheadVector,np.array((0,0,1)))))
            alpha = np.arctan2(lookaheadVector[0],lookaheadVector[1])
            # if (lookaheadVector[0] < 0):
            #     alpha *= -1.0
                #alphaVelocity *= -1.0
            physical_angle = np.arctan((2 * L_*np.sin(alpha)) / D)
            #physical_angle = np.arctan2(lookaheadVector[0],lookaheadVector[1])
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
            
            
            plt.draw()
            plt.pause(1e-17)
            time.sleep(0.025)
            
    except KeyboardInterrupt as e:
        controller.setControl(0.0,0.0,0.0)
        running = False
        sock.close()
        time.sleep(0.25)
    except Exception as e:
    #    global running, sock
        print(e)
        traceback.print_exc(file=sys.stdout)
        controller.setControl(0.0,0.0,0.0)
        running = False
        sock.close()
        time.sleep(0.25)
        exit(0)
    
    if (packet_queue is not None):
        print("Flushing %d more packets to disk." %(packet_queue.qsize()))
        while( not packet_queue.empty()):
            time.sleep(1)
    if (setpoint_queue is not None) and (actual_queue is not None):
        with open("setpoint_vels.txt","w") as f:
            strings = [str(setpoint_queue[i])+","+str(actual_queue[i])+"\n" for i in range(len(setpoint_queue))]
            f.writelines(strings)
if __name__ == '__main__':
    logging.basicConfig()
    serve()