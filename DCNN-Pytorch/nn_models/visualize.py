import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
import os
import pickle
import string
import argparse 
import nn_models.Models as models
import matplotlib.pyplot as plt
import cv2

def get_fmaps(network,input,layer=5):
    output,maps = network(input)
    cvt2pil = transforms.ToPILImage()
    plt.imshow(cvt2pil(maps[layer-1].squeeze().data.cpu()))
    plt.show()
    return

def load_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    return img

def main():
    parser = argparse.ArgumentParser(description="Visualize AdmiralNet")
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--layer", type=str, required=True)
    args = parser.parse_args()
    
    model_dir, model_file = os.path.split(args.model_file)
    config_path = os.path.join(model_dir,'config.pkl')
    config_file = open(config_path,'rb')
    config = pickle.load(config_file)
    print(config)

    gpu = int(config['gpu'])
    use_float32 = bool(config['use_float32'])
    label_scale = float(config['label_scale'])
    size = (66,200)
    context_length = int(config['context_length'])
    sequence_length = int(config['sequence_length'])
    hidden_dim = int(config['hidden_dim'])
    optical_flow = bool(config.get('optical_flow',''))
    
    if(optical_flow):
        prvs = load_image(args.input_file).astype(np.float32) / 255.0
        prvs_grayscale = cv2.cvtColor(prvs,cv2.COLOR_BGR2GRAY)
        prvs_resize = cv2.resize(prvs_grayscale, (self.im_size[1], self.im_size[0]), interpolation = cv2.INTER_CUBIC)
        flow = cv2.calcOpticalFlowFarneback(prvs_resize,next_resize, None, 0.5, 3, 20, 8, 5, 1.2, 0)
        inputfile=flow.transpose(2, 0, 1)
        inputfile=inputfile.reshape(-1,2,1825,300)    
    else:
        inputfile = load_image(args.input_file).astype(np.float32) / 255.0
        inputfile=inputfile.reshape(-1,3,1825,300)

    network = models.AdmiralNet(cell='lstm',context_length = context_length, sequence_length=sequence_length, hidden_dim = hidden_dim, use_float32 = use_float32, gpu = gpu, optical_flow=optical_flow)
    state_dict = torch.load(args.model_file)
    network.load_state_dict(state_dict)
    print(network)

    get_fmaps(network,inputfile,int(args.layer))

if __name__ == '__main__':
    main()




