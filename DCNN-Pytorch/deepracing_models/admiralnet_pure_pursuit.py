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
import deepracing.controls
import endtoend_controls.EndToEndPurePursuit
def serve():
    global velsetpoint, current_motion_data, throttle_out, running, speed
    parser = argparse.ArgumentParser(description='Pure Pursuit.')
    parser.add_argument('model_file', type=str)
    parser.add_argument('address', type=str)
    parser.add_argument('port', type=int)
    parser.add_argument('trackfile', type=str)
    parser.add_argument('--lookahead_gain', type=float, default=0.3, required=False)
    parser.add_argument('--pgain', type=float, default=1.0, required=False)
    parser.add_argument('--igain', type=float, default=0.0225, required=False)
    parser.add_argument('--dgain', type=float, default=0.0125, required=False)
    parser.add_argument('--vmax', type=float, default=175.0, required=False)
    parser.add_argument('--logdir', type=str, default=None, required=False)
    parser.add_argument('--usesplines', action="store_true")
    args = parser.parse_args()
    model_file = args.model_file
    address = args.address
    port = args.port
    trackfile = args.trackfile
    control = endtoend_controls.EndToEndPurePursuit.AdmiralNetPurePursuitController(model_file, trackfile, address=address, port=port, pgain=args.pgain, igain=args.igain, dgain=args.dgain, lookahead_gain=args.lookahead_gain)
    control.start()
    print("Cntrl-C to exit")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt as e:
        print("Thanks for playing!")
        control.stop()
    except Exception as e:
        print(e)
        control.stop()
if __name__ == '__main__':
    logging.basicConfig()
    serve()