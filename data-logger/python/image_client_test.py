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
from deepracing.backend.ImageBackends import ImageGRPCClient as ImageGRPCClient
print(os.environ['PYTHONPATH'])
parser = argparse.ArgumentParser(description='Image client.')
parser.add_argument('address', type=str)
parser.add_argument('port', type=int)
parser.add_argument('key', type=str)
args = parser.parse_args()
wrapper = ImageGRPCClient(address=args.address, port=args.port)
tick = time.time()
im = wrapper.getImage(args.key)
tock = time.time()
imviz = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
print( "Got image and reshaped in %f milliseconds." % ((tock-tick)*1000.0) )
cv2.namedWindow("ImageResponse",cv2.WINDOW_AUTOSIZE)
cv2.imshow("ImageResponse",imviz)
cv2.waitKey(0)