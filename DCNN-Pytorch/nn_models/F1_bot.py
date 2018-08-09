import numpy as np
from PIL import ImageGrab
import cv2
from directkeys import PressKey,ReleaseKey, W, A, S, D
import nn_models.Models as models
import torch
import torch.nn as nn 
import pickle
import os
import string
import argparse
import time

def right(angle):
    ReleaseKey(A)
    PressKey(D)
    time.sleep(2*(angle*angle))
    ReleaseKey(D)

def left(angle):
    ReleaseKey(D)
    PressKey(A)
    time.sleep(2*(angle*angle))
    ReleaseKey(A)
    
def straight():
    ReleaseKey(A)
    ReleaseKey(D)
    #time.sleep(0.05)
    

def grab_screen():
    screen =  np.array(ImageGrab.grab(bbox=(0,430,2510,630)))
    return screen

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
    print(network)
    
    throttle = torch.Tensor(1,10)
    brake = torch.Tensor(1,10)
    if(use_float32):
        network.float()
    else:
        network.double()
    if(gpu>=0):
        network = network.cuda(gpu)
    network.eval()

    time.sleep(3)
    inputs = []
    pscreen = grab_screen()
    pscreen = cv2.cvtColor(pscreen,cv2.COLOR_BGR2GRAY)
    pscreen = cv2.resize(pscreen,(200,66))
    buffer = 0
    while(True):
        while(buffer<context_length):
            screen = grab_screen()
            screen = cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen,(200,66))
            flow = cv2.calcOpticalFlowFarneback(pscreen,screen, None, 0.5, 3, 20, 8, 5, 1.2, 0)
            im= flow.transpose(2, 0, 1)
            #cv2.imwrite("screen"+str(buffer)+".jpeg",flow[...,1])
            inputs.append(im)
            buffer+=1
            time.sleep(0.04)
            pscreen = screen
        img_list = np.asarray(inputs)
        img_tensor = torch.from_numpy(img_list)
        img_tensor = img_tensor.view(-1,context_length,2,66,200)
        screen = grab_screen()
        screen = torch.from_numpy(screen)

        if(gpu>=0):
            img_tensor = img_tensor.cuda(gpu)
            throttle = throttle.cuda(gpu)
            brake = brake.cuda(gpu)
        pred = network(img_tensor,throttle,brake)
        if pred.shape[1] == 1:
            angle = pred.item()
        else:
            angle = torch.sum(pred.squeeze()).item()/float(context_length)
        #print(angle)
        inputs = inputs[1:]
        buffer -= 1
        if(angle>0.15):
            if(angle<0.6):
                right(angle)
            else:
                right((1+angle/2)*angle)
            print('D')
        elif(angle<-0.15):
            if(angle<-0.6):
                left(-1*(angle))
            else:
                left(-(1+angle/2)*(angle))
            print('A')
        else:
            straight()

if __name__ == '__main__':
    main()