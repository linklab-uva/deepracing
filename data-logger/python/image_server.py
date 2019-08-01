import ImageRPC_pb2_grpc
import ImageRPC_pb2
import Image_pb2
import ChannelOrder_pb2
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
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class ImageFileServer(ImageRPC_pb2_grpc.ImageServiceServicer):
  def __init__(self, image_folder : str, resize_factor : float = 1.0):
    super(ImageFileServer, self).__init__()
    self.image_folder=image_folder
    self.resize_factor=resize_factor
  def GetImage(self, request, context):
    imcv = cv2.resize( cv2.imread( os.path.join(self.image_folder, request.key) , cv2.IMREAD_UNCHANGED ) , None , fx=self.resize_factor , fy=self.resize_factor , interpolation=cv2.INTER_AREA )
    rtn =  ImageRPC_pb2.ImageResponse()
    rtn.image.channel_order = ChannelOrder_pb2.ChannelOrder.BGR
    rtn.image.rows = imcv.shape[0]
    rtn.image.cols = imcv.shape[1]
    rtn.image.image_data = imcv.flatten().tobytes()
    return rtn

def serve():
    parser = argparse.ArgumentParser(description='Image server.')
    parser.add_argument('address', type=str)
    parser.add_argument('port', type=int)
    parser.add_argument('image_folder', type=str)
    parser.add_argument('resize_factor', type=float)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.num_workers))
    lmdbserver = ImageFileServer(args.image_folder, resize_factor=args.resize_factor)
    ImageRPC_pb2_grpc.add_ImageServiceServicer_to_server(lmdbserver, server)
    server.add_insecure_port('%s:%d' % (args.address,args.port) )
    print("Starting server")
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    logging.basicConfig()
    serve()