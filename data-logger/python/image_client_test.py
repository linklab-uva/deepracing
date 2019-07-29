import ImageRPC_pb2_grpc
import ImageRPC_pb2
import grpc
import cv2
import numpy as np
import argparse
import skimage
import skimage.io as io
from skimage.viewer import ImageViewer
import os
print(os.environ['PYTHONPATH'])
parser = argparse.ArgumentParser(description='Image client.')
parser.add_argument('address', type=str)
parser.add_argument('port', type=int)
parser.add_argument('key', type=str)
args = parser.parse_args()
channel = grpc.insecure_channel("%s:%d" %(args.address, args.port))
stub = ImageRPC_pb2_grpc.ImageServiceStub(channel)
response = stub.GetImage(ImageRPC_pb2.ImageRequest(key=args.key))
imshape = np.array((response.image.rows, response.image.cols, 3))
im = np.reshape(np.frombuffer(response.image.image_data,dtype=np.uint8), imshape)
viewer = ImageViewer(im)
print(viewer.show())