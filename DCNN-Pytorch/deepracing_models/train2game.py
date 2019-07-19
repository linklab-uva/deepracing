import numpy as np
import py_vjoy
import argparse
from PIL import ImageGrab
import cv2
import nn_models.Models as models
import torch
import torch.nn as nn 
import pickle
import os
import string
import time
import pyf1_datalogger
import torchvision.transforms as transforms
import imutils.annotation_utils
import numpy_ringbuffer
def grab_screen(dl):
    return dl.read()
def fill_buffer(buffer, dataloader, dt=0.0, context_length=10, interp = cv2.INTER_AREA):
    pscreen = grab_screen(dataloader)
    pscreen = cv2.cvtColor(pscreen,cv2.COLOR_BGR2GRAY)
    pscreen = cv2.resize(pscreen,(200,66), interpolation=interp)
    while(buffer.shape[0]<context_length):
        cv2.waitKey(dt)
        screen = grab_screen(dataloader)
        screen = cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen,(200,66), interpolation=interp)
        flow = cv2.calcOpticalFlowFarneback(pscreen,screen, None, 0.5, 3, 20, 8, 5, 1.2, 0)
        im= flow.transpose(2, 0, 1).astype(np.float32)
        buffer.append(im)
        pscreen = screen
    return pscreen
def main():
    parser = argparse.ArgumentParser(description="Test AdmiralNet")
    parser.add_argument("--model_file", type=str, required=True)
    args = parser.parse_args()
    
    model_dir, model_file = os.path.split(args.model_file)
    config_path = os.path.join(model_dir,'config.pkl')
    config_file = open(config_path,'rb')
    config = pickle.load(config_file)
    model_prefix, _ = model_file.split(".")

    gpu = int(config['gpu'])
    use_float32 = bool(config['use_float32'])
    label_scale = float(config['label_scale'])
    context_length = int(config['context_length'])
    sequence_length = int(config['sequence_length'])
    hidden_dim = int(config['hidden_dim'])
    optical_flow = bool(config.get('optical_flow',''))
    rnn_cell_type='lstm'
    network = models.AdmiralNet(cell=rnn_cell_type,context_length = context_length, sequence_length=sequence_length, hidden_dim = hidden_dim, use_float32 = use_float32, gpu = gpu, optical_flow=optical_flow)
    state_dict = torch.load(args.model_file)
    network.load_state_dict(state_dict)
    network=network.float()
    network=network.cuda(0)
    print(network)
    vjoy_max = 32000
    
    throttle = torch.Tensor(1,10)
    brake = torch.Tensor(1,10)
    if(use_float32):
        network.float()
    else:
        network.double()
    if(gpu>=0):
        network = network.cuda(gpu)
    network.eval()
    vj = py_vjoy.vJoy()
    vj.capture(1) #1 is the device ID
    vj.reset()
    js = py_vjoy.Joystick()
    js.setAxisXRot(int(round(vjoy_max/2))) 
    js.setAxisYRot(int(round(vjoy_max/2))) 
    vj.update(js)
    time.sleep(2)
    inputs = []
    '''
    '''
    wheel_pred = cv2.imread('predicted_fixed.png',cv2.IMREAD_UNCHANGED)
    wheelrows_pred = 66
    wheelcols_pred = 66
    wheel_pred = cv2.resize(wheel_pred, (wheelcols_pred,wheelrows_pred), interpolation = cv2.INTER_CUBIC)
    buffer = numpy_ringbuffer.RingBuffer(capacity=context_length, dtype=(np.float32, (2,66,200) ) )

    dt = 12
    context_length=10
    debug=True
    app="F1 2017"
    dl = pyf1_datalogger.ScreenVideoCapture()
    dl.open(app,0,200,1700,300)
    interp = cv2.INTER_AREA
    if debug:
        cv2.namedWindow(app, cv2.WINDOW_AUTOSIZE)
    pscreen = fill_buffer(buffer,dl,dt=dt,context_length=context_length,interp=interp)
    buffer_torch = torch.rand(1,10,2,66,200).float()
    buffer_torch=buffer_torch.cuda(0)
    while(True):
        cv2.waitKey(dt)
        screen = grab_screen(dl)
        screen_grey = cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY)
        screen_grey = cv2.resize(screen_grey,(200,66), interpolation=interp)
        flow = cv2.calcOpticalFlowFarneback(pscreen,screen_grey, None, 0.5, 3, 20, 8, 5, 1.2, 0)
        im= flow.transpose(2, 0, 1).astype(np.float32)
        buffer.append(im)
        pscreen = screen_grey
        buffer_torch[0] = torch.from_numpy(np.array(buffer))
        #print("Input Size: " + str(buffer_torch.size()))
        outputs = network(buffer_torch, throttle=None, brake=None )
        angle = outputs[0][0].item()
        print("Output: " + str(angle))
        scaled_pred_angle = 180.0*angle+7
        M_pred = cv2.getRotationMatrix2D((wheelrows_pred/2,wheelcols_pred/2),scaled_pred_angle,1)
        wheel_pred_rotated = cv2.warpAffine(wheel_pred,M_pred,(wheelrows_pred,wheelcols_pred))
        background = screen
        out_size = background.shape
        print(out_size)
        print(wheel_pred_rotated.shape)
        overlayed_pred = imutils.annotation_utils.overlay_image(background,wheel_pred_rotated,int((out_size[1]-wheelcols_pred)/2),int((out_size[0]-wheelcols_pred)/2))
        if debug:
            cv2.imshow(app,overlayed_pred)
        vjoy_angle = -angle*vjoy_max + vjoy_max/2.0
        js.setAxisXRot(int(round(vjoy_angle))) 
        js.setAxisYRot(int(round(vjoy_angle))) 
        vj.update(js)
        '''
        '''
        
    print(buffer.shape)             
        
if __name__ == '__main__':
    main()