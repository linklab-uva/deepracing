import DeepF1_RPC_pb2_grpc
import DeepF1_RPC_pb2
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
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

def pbImageToNpImage(im_pb : Image_pb2.Image):
    im = None
    if im_pb.channel_order == ChannelOrder_pb2.BGR:
        im = np.reshape(np.frombuffer(im_pb.image_data,dtype=np.uint8),np.array((im_pb.rows, im_pb.cols, 3)))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    elif im_pb.channel_order == ChannelOrder_pb2.RGB:
        im = np.reshape(np.frombuffer(im_pb.image_data,dtype=np.uint8),np.array((im_pb.rows, im_pb.cols, 3)))
    elif im_pb.channel_order == ChannelOrder_pb2.GRAYSCALE:
        im = np.reshape(np.frombuffer(im_pb.image_data,dtype=np.uint8),np.array((im_pb.rows, im_pb.cols)))
    elif im_pb.channel_order == ChannelOrder_pb2.RGBA:
        im = np.reshape(np.frombuffer(im_pb.image_data,dtype=np.uint8),np.array((im_pb.rows, im_pb.cols, 4)))
    elif im_pb.channel_order == ChannelOrder_pb2.BGRA:
        im = np.reshape(np.frombuffer(im_pb.image_data,dtype=np.uint8),np.array((im_pb.rows, im_pb.cols, 4)))
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError("Unknown channel order: " + im_pb.channel_order)
    return im
class ImageHandler(DeepF1_RPC_pb2_grpc.SendImageServicer):
  def __init__(self, imsize, context_length):
    super(ImageHandler, self).__init__()
    self.imsize = imsize
    self.ring_buffer = RingBuffer(context_length,dtype=(np.uint8, (imsize[0],imsize[1],3) ) )
    for i in range(context_length):
        self.ring_buffer.append(np.zeros((imsize[0],imsize[1],3),dtype=np.uint8))
  def SendImage(self, request, context):
    im = pbImageToNpImage(request.impb)
    self.ring_buffer.append(im[:,:,0:3])
    res = DeepF1_RPC_pb2_grpc.DeepF1__RPC__pb2.SendImageResponse()
    res.err="Hello World!"
    return res
def serve():
    parser = argparse.ArgumentParser(description='Image server.')
    parser.add_argument('address', type=str)
    parser.add_argument('port', type=int)
    parser.add_argument('modelfile', type=str)
    args = parser.parse_args()
    modelfile = args.modelfile

    modeldir = os.path.dirname(modelfile)
    cfgfile = os.path.join(modeldir,"config.yaml")
    config = yaml.load(open(cfgfile,"r"), Loader=yaml.SafeLoader)
    image_size = config["image_size"]
    hidden_dimension = config["hidden_dimension"]
    input_channels = config["input_channels"]
    sequence_length = config["sequence_length"]
    context_length = config["context_length"]
    gpu = 1
    temporal_conv_feature_factor = config["temporal_conv_feature_factor"]
    position_loss_reduction = config["position_loss_reduction"]
    use_float = config["use_float"]
    learnable_initial_state = config.get("learnable_initial_state",False)

    net = AdmiralNetPosePredictor(gpu=gpu,context_length = context_length, sequence_length = sequence_length,\
        hidden_dim=hidden_dimension, input_channels=input_channels, temporal_conv_feature_factor = temporal_conv_feature_factor, \
            learnable_initial_state =learnable_initial_state)
    net.load_state_dict(torch.load(modelfile,map_location=torch.device("cpu")))
    net = net.double()
    if gpu>=0:
        net = net.cuda(gpu)

    server = grpc.server( futures.ThreadPoolExecutor(max_workers=1) )
    handler = ImageHandler(image_size,context_length)
    server.add_insecure_port('%s:%d' % (args.address,args.port) )
    DeepF1_RPC_pb2_grpc.add_SendImageServicer_to_server(handler, server)
    print("Starting image server")
    server.start()
    totens = tf.ToTensor()
    inp = input("Enter anything to continue\n")
    time.sleep(3)
    # windowname = "Test Image"
    # cv2.namedWindow(windowname,cv2.WINDOW_AUTOSIZE)
    # x_,y_,w_,h_ = cv2.selectROI(windowname, cv2.cvtColor(np.array(handler.ring_buffer)[0], cv2.COLOR_RGB2BGR), showCrosshair =True)
    # #print((x_,y_,w_,h_))
    # x = int(round(x_))
    # y = int(round(y_))
    # w = int(round(w_))
    # h = int(round(h_))
    # print((x,y,w,h))
    try:
        cv2.namedWindow("imrecv", cv2.WINDOW_AUTOSIZE)
        net = net.eval()
        mask = 2.0*torch.ones(sequence_length,3).double()
        if gpu>=0:
            mask = mask.cuda(gpu)
        while True:
            input_np = np.array(handler.ring_buffer)
            if input_np.shape[0]<context_length:
                continue
            input_torch = torch.from_numpy(np.array([totens(input_np[i]).numpy() for i in range(context_length)])).unsqueeze(0)
            input_torch = input_torch.double()
            input_torch.requires_grad=False
            if gpu>=0:
                input_torch = input_torch.cuda(gpu)
            #print(input_torch.shape)
            output = net(input_torch)[0]
            output[torch.abs(output)<mask]=0.0
            print(output[0:15])
            cv2.imshow("imrecv", cv2.cvtColor(np.array(input_np[context_length-1]),cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
    except KeyboardInterrupt:
        server.stop(0)
  
if __name__ == '__main__':
    logging.basicConfig()
    serve()