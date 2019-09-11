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
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
current_motion_data = None
error_ring_buffer = RingBuffer( 10, dtype=np.float64 )
throttle_out = 0.0
dt = 1/60.0
t = dt*np.linspace(0,9,10)
speed = 0.0
running = True
velsetpoint = 0.0
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
class MotionDataHandler(DeepF1_RPC_pb2_grpc.SendMotionDataServicer):
  def __init__(self):
    super(MotionDataHandler, self).__init__()
  def SendMotionData(self, request, context):
    global current_motion_data
    res = DeepF1_RPC_pb2_grpc.DeepF1__RPC__pb2.SendMotionDataResponse()
    current_motion_data = request.motion_data
    return res
class ImageHandler(DeepF1_RPC_pb2_grpc.SendImageServicer):
  def __init__(self, imsize, context_length):
    super(ImageHandler, self).__init__()
    self.imsize = imsize
    self.imprev = None
    self.imcurr = None
    self.ring_buffer = RingBuffer(context_length,dtype=(np.uint8, (imsize[0],imsize[1],3) ) )
    self.flow_ring_buffer = RingBuffer(context_length,dtype=(np.float32, (imsize[0],imsize[1],2) ) )
    for i in range(context_length):
        self.ring_buffer.append(np.zeros((imsize[0],imsize[1],3),dtype=np.uint8))
  def SendImage(self, request, context):
    self.imcurr = pbImageToNpImage(request.impb)[:,:,0:3]
    self.ring_buffer.append(self.imcurr)
    if (self.imprev is not None ) and (self.imcurr is not None):
        try:
            img_prev_grey = cv2.cvtColor(self.imprev,cv2.COLOR_RGB2GRAY)
            img_curr_grey = cv2.cvtColor(self.imcurr,cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(img_prev_grey, img_curr_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0).astype(np.float32)
            self.flow_ring_buffer.append(flow)
        except:
            print("Error computing optical flow")
    self.imprev = self.imcurr#.copy()
    res = DeepF1_RPC_pb2_grpc.DeepF1__RPC__pb2.SendImageResponse()
    res.err="Hello World!"
    return res
def serve():
    global velsetpoint 
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

    motion_data_server = grpc.server( futures.ThreadPoolExecutor(max_workers=1) )
    motion_data_handler = MotionDataHandler()
    motion_data_server.add_insecure_port('%s:%d' % (args.address,args.port+1) )
    DeepF1_RPC_pb2_grpc.add_SendMotionDataServicer_to_server(motion_data_handler, motion_data_server)
    print("Starting motion data server")
    motion_data_server.start()


    totens = tf.ToTensor()
    #2.237
    vmax = 65.0/2.237
    velsetpoint = vmax
    inp = input("Enter anything to continue\n")
    time.sleep(2.0)
    vel_control_thread = threading.Thread(target=velocityControl, args=(1.0, 1E-5))
    vel_control_thread.start()
    # windowname = "Test Image"
    # cv2.namedWindow(windowname,cv2.WINDOW_AUTOSIZE)
    # x_,y_,w_,h_ = cv2.selectROI(windowname, cv2.cvtColor(np.array(handler.ring_buffer)[0], cv2.COLOR_RGB2BGR), showCrosshair =True)
    # #print((x_,y_,w_,h_))
    # x = int(round(x_))
    # y = int(round(y_))
    # w = int(round(w_))
    # h = int(round(h_))
    # print((x,y,w,h))
    lookahead_indices = sequence_length-10
    controller = py_f1_interface.F1Interface(1)
    controller.setControl(0.0,0.0,0.0)
    L_ = 3.629
    lookahead_gain = 0.2
    lookahead_gain_vel = 1.75
    try:
        cv2.namedWindow("imrecv", cv2.WINDOW_AUTOSIZE)
        net = net.eval()
        mask = 0.05*torch.ones(sequence_length,3).double()
        input_torch = torch.zeros(1,context_length,net.input_channels,image_size[0],image_size[1]).double()
        input_torch.requires_grad=False
        t_interp = np.linspace(0,1.0,lookahead_indices)
        if gpu>=0:
            input_torch = input_torch.cuda(gpu)
            mask = mask.cuda(gpu)
        smin = .10
        while True:
            global throttle_out
            time.sleep(0.0025)
            input_np = np.array(handler.ring_buffer)
            flow_np = np.array(handler.flow_ring_buffer)
            if not ((input_np.shape[0]==context_length) and (flow_np.shape[0]==context_length)):
                continue
            if gpu>=0:
                if net.input_channels==5:
                   # print("getting optical flow tensor")
                    flowtens = torch.from_numpy(np.array([flow_np[i].transpose(2,0,1) for i in range(context_length)])).double().cuda(gpu)
                   # print("concatenating tensors")
                    input_torch[0] = torch.cat((torch.from_numpy(np.array([totens(input_np[i]).numpy() for i in range(context_length)])).double().cuda(gpu) ,\
                         flowtens ), dim=1)
                else:
                    input_torch[0] = torch.from_numpy(np.array([totens(input_np[i]).numpy() for i in range(context_length)])).double().cuda(gpu)
            else:
                if gpu>=0:
                    if net.input_channels==5:
                        input_torch[0] = torch.cat((torch.from_numpy(np.array([totens(input_np[i]).numpy() for i in range(context_length)])).double() ,\
                            torch.from_numpy(np.array([flow_np[i].transpose(2,0,1) for i in range(context_length)])).double().cuda(gpu) ), dim=1)
                    else:
                        input_torch[0] = torch.from_numpy(np.array([totens(input_np[i]).numpy() for i in range(context_length)])).double()
            #print(input_torch.shape)
            output = net(input_torch)[0][0]
            output[torch.abs(output)<mask]=0.0
            #print(output.shape)
            output_np = output[0:lookahead_indices].detach().cpu().numpy()
            spline_np = scipy.interpolate.interp1d(t_interp, output_np , axis=0, kind='linear')
            lookahead_t = lookahead_gain*(speed/vmax)
            lookahead_t_vel = lookahead_gain_vel*(speed/vmax)
            if lookahead_t>1.0:
                lookahead_t = 1.0
            elif lookahead_t < smin:
                lookahead_t = smin

            if lookahead_t_vel>1.0:
                lookahead_t_vel = 1.0
            elif lookahead_t_vel < smin:
                lookahead_t_vel = smin
            delta = 0.0
           # print(lookahead_t)
            looakhead_point = spline_np(lookahead_t)
            looakhead_point[2]+=1.5
            looakhead_point_vel = spline_np(lookahead_t_vel)
           # print("lookahead_t: %f" % lookahead_t)
           # print(looakhead_point)
            D = la.norm(looakhead_point)
            Dvel = la.norm(looakhead_point_vel)
            lookaheadVector = looakhead_point/D
            lookaheadVectorVel = looakhead_point_vel/Dvel
            alpha = np.abs(np.arccos(np.dot(lookaheadVector,np.array((0.0,0.0,1.0)))))
            alphavel = np.abs(np.arccos(np.dot(lookaheadVectorVel,np.array((0.0,0.0,1.0)))))
            velsetpoint = max(vmax*((1.0-(alphavel/1.57))**3), 35)
            if lookaheadVector[0]<0.0:
                alpha *= -1.0
            #print("alpha calc: %f" % alpha)
            print("vel setpoint: %f" % velsetpoint)
            physical_angle = -1.0* np.arctan((2 * L_*np.sin(alpha)) / la.norm(looakhead_point))
            if physical_angle > 0 :
                delta = -3.79616039*physical_angle# + 0.01004506
            else:
                delta = -3.34446413*physical_angle# + 0.01094534
            if abs(delta)>0.05:
                delta=1.2*delta
            else:
                delta=1.0*delta
            print("Delta out: %f" % delta)
            if throttle_out>0:
                controller.setControl(delta,throttle_out,0.0)
            else:
                controller.setControl(delta,0.0,-throttle_out)
            #if(current_motion_data is not None):
              #  vel = deepracing.pose_utils.extractVelocity(current_motion_data,0)
                #print(vel)
            cv2.imshow("imrecv", cv2.cvtColor(np.round(255.0*input_torch[0][context_length-1][0:3].cpu().numpy().transpose(1,2,0)).astype(np.uint8),cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
    except Exception as e:
        global running
        controller.setControl(0.0,0.0,0.0)
        cv2.destroyWindow("imrecv")
        print(e)
        running = False
        server.stop(0)
        motion_data_server.stop(0)
        exit(0)
  
if __name__ == '__main__':
    logging.basicConfig()
    serve()