import comet_ml
import torch
import torch.nn as NN
import torch.nn.functional as F
import torch.utils.data as data_utils
import deepracing_models.data_loading.proto_datasets as PD
from tqdm import tqdm as tqdm
import deepracing_models.nn_models.LossFunctions as loss_functions
import deepracing_models.nn_models.Models
import numpy as np
import torch.optim as optim
import pickle
from datetime import datetime
import os
import string
import argparse
import torchvision.transforms as transforms
import yaml
import shutil
import skimage
import skimage.io
import deepracing
from deepracing import trackNames
import deepracing.backend
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import deepracing_models.math_utils.bezier
import socket
import json
from comet_ml.api import API, APIExperiment
import cv2
import torchvision, torchvision.transforms as T
from deepracing_models.data_loading.image_transforms import GaussianBlur

#torch.backends.cudnn.enabled = False
def runtest(network, dataloader, ego_agent_loss, use_tqdm = False, debug=False):
    cum_loss = 0.0
    cum_param_loss = 0.0
    cum_position_loss = 0.0
    cum_velocity_loss = 0.0
    num_samples=0.0
    dataloaderlen = len(dataloader)
    if use_tqdm:
        t = tqdm(enumerate(dataloader), total=dataloaderlen)
    else:
        t = enumerate(dataloader)
    network.eval()
    dev = next(network.parameters()).device  # we are only doing single-device for now, so this is fine.
    fix_first_point = config["fix_first_point"]
    loss_weights = config["loss_weights"]

    bezier_order = network.params_per_dimension-1+int(fix_first_point)

    for (i, imagedict) in t:
        input_images = imagedict["images"].double().to(device=dev)
        ego_current_pose = imagedict["ego_current_pose"].double().to(device=dev)
        session_times = imagedict["session_times"].double().to(device=dev)
        ego_positions = imagedict["ego_positions"].double().to(device=dev)
        ego_velocities = imagedict["ego_velocities"].double().to(device=dev)
        batch_size = input_images.shape[0]
        
        
        predictions = network(input_images)
        if fix_first_point:
            initial_zeros = torch.zeros(batch_size,1,2,dtype=torch.float64,device=dev)
            network_output_reshape = predictions.transpose(1,2)
            predictions_reshape = torch.cat((initial_zeros,network_output_reshape),dim=1)
        else:
            predictions_reshape = predictions.transpose(1,2)

        dt = session_times[:,-1]-session_times[:,0]
        s_torch_cur = (session_times - session_times[:,0,None])/dt[:,None]

        Mpos = deepracing_models.math_utils.bezierM(s_torch_cur, bezier_order)

        pred_points = torch.matmul(Mpos, predictions_reshape)


        current_loss = ego_agent_loss(pred_points, ego_positions)

        if use_tqdm:
            t.set_postfix({"current_loss" : current_loss.item()})
def go():
    parser = argparse.ArgumentParser(description="Test AdmiralNet Bezier Curve Predictor")
    parser.add_argument("dataset_file", type=str,  help="Dataset Configuration file to load")
    parser.add_argument("model_file", type=str,  help="Model Configuration file to load")

    parser.add_argument("--debug", action="store_true",  help="Display images upon each iteration of the training loop")
    parser.add_argument("--tqdm", action="store_true",  help="Display tqdm progress bar on each epoch")
    parser.add_argument("--gpu", type=int, default=None,  help="Override the GPU index specified in the config file")

    
    args = parser.parse_args()
    argdict = vars(args)
    use_tqdm = argdict["tqdm"]


    dataset_file = argdict["dataset_file"]
    with open(dataset_config_file,"r") as f:
        dataset_config = yaml.load(f, Loader = yaml.SafeLoader)
    
    model_file = argdict["model_file"]
    config_file = os.path.join(os.path.dirname(model_file),"model_config.yaml")
    with open(config_file,"r") as f:
        model_config = yaml.load(f, Loader = yaml.SafeLoader)

   

    context_length = model_config["context_length"]
    input_channels = model_config["input_channels"]
    hidden_dim = model_config["hidden_dimension"]
    use_3dconv = model_config["use_3dconv"]
    bezier_order = model_config["bezier_order"]
    fix_first_point = model_config["fix_first_point"]
    num_recurrent_layers = model_config.get("num_recurrent_layers",1)


    
    net = deepracing_models.nn_models.Models.AdmiralNetCurvePredictor( context_length = context_length , input_channels=input_channels, hidden_dim = hidden_dim, num_recurrent_layers=num_recurrent_layers, params_per_dimension=bezier_order + 1 - int(fix_first_point), use_3dconv = use_3dconv) 

    loss = deepracing_models.nn_models.LossFunctions.SquaredLpNormLoss()

    print("casting stuff to double")
    net = net.double()
    ego_agent_loss = ego_agent_loss.double()
    other_agent_loss = other_agent_loss.double()

    if model_load is not None:
        net.load_state_dict(torch.load(model_load, map_location=torch.device("cpu")))
    if gpu>=0:
        print("moving stuff to GPU")
        net = net.cuda(gpu)
        loss = loss.cuda(gpu)
    else:
        device = torch.device("cpu")
    
    
    dsets=[]
    dsetfolders = []
    alltags = set(dataset_config.get("tags",[]))
    dset_output_lengths=[]
    return_other_agents = bool(dataset_config.get("other_agents",False))
    for dataset in dataset_config["datasets"]:
        dlocal : dict = {k: dataset_config[k] for k in dataset_config.keys()  if (not (k in ["datasets"]))}
        dlocal.update(dataset)
        print("Parsing database config: %s" %(str(dlocal)))
        key_file = dlocal["key_file"]
        root_folder = dlocal["root_folder"]
        position_indices = dlocal["position_indices"]
        label_subfolder = dlocal["label_subfolder"]
        dataset_tags = dlocal.get("tags", [])
        alltags = alltags.union(set(dataset_tags))

        dsetfolders.append(root_folder)
        label_folder = os.path.join(root_folder,label_subfolder)
        image_folder = os.path.join(root_folder,"images")
        key_file = os.path.join(root_folder,key_file)
        label_wrapper = deepracing.backend.MultiAgentLabelLMDBWrapper()
        label_wrapper.openDatabase(os.path.join(label_folder,"lmdb") )


        image_mapsize = float(np.prod(image_size)*3+12)*float(len(label_wrapper.getKeys()))*1.1
        image_wrapper = deepracing.backend.ImageLMDBWrapper()
        image_wrapper.readDatabase( os.path.join(image_folder,"image_lmdb"), mapsize=image_mapsize )

        extra_transforms = []
        color_jitters = dlocal.get("color_jitters", None) 
        if color_jitters is not None:
            extra_transforms+=[T.ColorJitter(brightness=[cj, cj]) for cj in color_jitters]
            
        blur = dlocal.get("blur", None)   
        if blur is not None:
            extra_transforms.append(GaussianBlur(blur))
        
        current_dset = PD.MultiAgentDataset(image_wrapper, label_wrapper, key_file, context_length, image_size, position_indices, extra_transforms=extra_transforms, return_other_agents=return_other_agents)
        dsets.append(current_dset)
        
        print("\n")
    if len(dsets)==1:
        dset = dsets[0]
    else:
        dset = torch.utils.data.ConcatDataset(dsets)
    
    dataloader = data_utils.DataLoader(dset, batch_size=1, shuffle=False, pin_memory=(gpu>=0))

import logging
if __name__ == '__main__':
    logging.basicConfig()
    go()    
    