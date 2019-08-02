import DeepF1_RPC_pb2_grpc
import DeepF1_RPC_pb2
import ChannelOrder_pb2
import grpc
import cv2
import numpy as np
import argparse
import skimage
import skimage.io as io
from skimage.viewer import ImageViewer
import os
import time
import deepracing.backend
import deepracing.backend 
import google.protobuf.empty_pb2
print(os.environ['PYTHONPATH'])
parser = argparse.ArgumentParser(description='Image client.')
parser.add_argument('address', type=str)
parser.add_argument('image_port', type=int)
parser.add_argument('label_port', type=int)
parser.add_argument('key', type=str)
args = parser.parse_args()
#channel = grpc.insecure_channel( "%s:%d" % ( args.address, args.image_port ) )
#stub = DeepF1_RPC_pb2_grpc.ImageServiceStub(channel)
image_wrapper =  deepracing.backend.ImageGRPCClient(address=args.address, port=args.image_port)
label_wrapper =  deepracing.backend.PoseSequenceLabelGRPCClient(address=args.address, port=args.label_port)
tick = time.time()
#im_pb = stub.GetImage(DeepF1_RPC_pb2.ImageRequest(key=args.key))
#metadata_pb = stub.GetDbMetadata(google.protobuf.empty_pb2.Empty())
im = image_wrapper.getImage(args.key)
label = label_wrapper.getPoseSequenceLabel(args.key)
tock = time.time()
print( label )
print("There are %d images in the database." %(image_wrapper.getNumImages()))
print("There are %d labels in the database." %(label_wrapper.getNumLabels()))
print( "Got image in %f milliseconds." % ((tock-tick)*1000.0) )

cv2.namedWindow("ImageResponse",cv2.WINDOW_AUTOSIZE)
cv2.imshow("ImageResponse",cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
cv2.waitKey(0)