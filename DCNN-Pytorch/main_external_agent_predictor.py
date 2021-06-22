import comet_ml
import torch
import torch.nn as NN
import torch.nn.functional as F
import torch.utils.data as data_utils
import deepracing_models.data_loading.proto_datasets as PD
from tqdm import tqdm as tqdm
import deepracing_models.nn_models.LossFunctions as loss_functions
from deepracing_models.nn_models.StateEstimationModels import ExternalAgentCurvePredictor
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
import deepracing_models.math_utils as mu
import socket
import json
from comet_ml.api import API, APIExperiment
import cv2
import torchvision, torchvision.transforms as T
from deepracing_models.data_loading.image_transforms import GaussianBlur
from deepracing.raceline_utils import loadBoundary
from deepracing import searchForFile
import deepracing.path_utils.geometric as geometric

#torch.backends.cudnn.enabled = False
def run_epoch(experiment : comet_ml.Experiment, network : ExternalAgentCurvePredictor, optimizer : torch.optim.Optimizer, dataloader : data_utils.DataLoader, debug : bool =False):
    num_samples=0.0
    t = tqdm(enumerate(dataloader), total=len(dataloader))
    network.train()  # This is important to call before training!
    
    # we are only doing single-device training for now, so this works fine.
    dev = next(network.parameters()).device
    dtype = next(network.parameters()).dtype 
    bezier_order = network.bezier_order
    lossf=0
    for (i, datadict) in t:
        valid_mask = datadict["valid_mask"] 
        past_positions = datadict["past_positions"]
        past_velocities = datadict["past_velocities"]
        future_positions = datadict["future_positions"]
        tfuture = datadict["tfuture"]

        valid_past_positions = (past_positions[valid_mask].double().cuda(0))[:,:,[0,2]]
        valid_past_velocities = (past_velocities[valid_mask].double().cuda(0))[:,:,[0,2]]
        valid_future_positions = (future_positions[valid_mask].double().cuda(0))[:,:,[0,2]]
        valid_tfuture = tfuture[valid_mask].double().cuda(0)

        networkinput = torch.cat([valid_past_positions, valid_past_velocities], dim=2)
        output = network(networkinput)
        if debug:
            pass

        dt = valid_tfuture[:,-1]-valid_tfuture[:,0]
        s_torch_cur = (valid_tfuture - valid_tfuture[:,0,None])/dt[:,None]
        Mpos = mu.bezierM(s_torch_cur, network.bezier_order)
        pred_points = torch.matmul(Mpos, output)
        deltas = pred_points - valid_future_positions
        squared_norms = torch.sum(torch.square(deltas), dim=2)
        loss = torch.mean(squared_norms)

        optimizer.zero_grad()
        loss.backward() 
        # Weight and bias updates.
        optimizer.step()
        # logging information
        lossf += loss.item()
        t.set_postfix({"current_position_loss" : lossf/(i+1)})
def go():
    parser = argparse.ArgumentParser(description="Train AdmiralNet Pose Predictor")
    parser.add_argument("training_config_file", type=str,  help="Training Configuration file to load")
    parser.add_argument("model_config_file", type=str,  help="Model Configuration file to load")
    parser.add_argument("output_directory", type=str,  help="Where to put the resulting model files")

    parser.add_argument("--debug", action="store_true",  help="Don't actually push to comet, just testing")
    parser.add_argument("--gpu", type=int, default=None,  help="Override the GPU index specified in the config file")

    
    args = parser.parse_args()
    argdict = vars(args)

    training_config_file = argdict["training_config_file"]
    model_config_file = argdict["model_config_file"]
    debug = argdict["debug"]

    with open(training_config_file) as f:
        training_config = yaml.load(f, Loader = yaml.SafeLoader)
    with open(model_config_file) as f:
        model_config = yaml.load(f, Loader = yaml.SafeLoader)
    
    bezier_order = model_config["bezier_order"]
    input_dim = model_config["input_dim"]
    output_dim = model_config["output_dim"]
    hidden_dim = model_config["hidden_dim"]
    num_layers = model_config["num_layers"]
    dropout = model_config["dropout"]
    bidirectional = model_config["bidirectional"]

    batch_size = training_config["batch_size"]
    learning_rate = training_config["learning_rate"]
    momentum = training_config["momentum"]
    nesterov = training_config["nesterov"]
    project_name = training_config["project_name"]
    num_epochs = training_config["num_epochs"]
    num_workers = training_config["num_workers"]
   
    if args.gpu is not None:
        gpu = args.gpu
        training_config["gpu"]  = gpu
    else:
        gpu = training_config["gpu"] 
    torch.cuda.set_device(gpu)    
    print("Using model config:\n%s" % (str(model_config)))
    net = ExternalAgentCurvePredictor(output_dim=output_dim, bezier_order=bezier_order, input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional) 
    print("net:\n%s" % (str(net)))
    net = net.float()
    
    if gpu>=0:
        print("moving stuff to GPU")
        net = net.cuda(gpu)
    optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum=momentum, nesterov=nesterov)

    dsets=[]
    for dataset in training_config["datasets"]:
        root_folder = dataset["root_folder"]
        current_dset = PD.PoseVelocityDataset(root_folder)
        dsets.append(current_dset)
    if len(dsets)==1:
        dset = dsets[0]
    else:
        dset = torch.utils.data.ConcatDataset(dsets)
    
    dataloader = data_utils.DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(gpu>=0))

    print("Dataloader of of length %d" %(len(dataloader)))
    if debug:
        print("Using datasets:\n%s", (str(training_config["datasets"])))
    
    main_dir = args.output_directory

    experiment = comet_ml.Experiment(workspace="electric-turtle", project_name=project_name)
    output_directory = os.path.join(main_dir, experiment.get_key())
    if os.path.isdir(output_directory) :
        raise FileExistsError("%s already exists, this should not happen." %(output_directory) )
    os.makedirs(output_directory)
    experiment.log_parameters(training_config)
    experiment.log_parameters(model_config)
    experiment_config = {"experiment_key": experiment.get_key()}
    yaml.dump(experiment_config, stream=open(os.path.join(output_directory,"experiment_config.yaml"),"w"), Dumper=yaml.SafeDumper)
    yaml.dump(training_config, stream=open(os.path.join(output_directory,"training_config.yaml"), "w"), Dumper = yaml.SafeDumper)
    yaml.dump(model_config, stream=open(os.path.join(output_directory,"model_config.yaml"), "w"), Dumper = yaml.SafeDumper)
    experiment.log_asset(os.path.join(output_directory,"experiment_config.yaml"),file_name="experiment_config.yaml")
    experiment.log_asset(os.path.join(output_directory,"training_config.yaml"),file_name="training_config.yaml")
    experiment.log_asset(os.path.join(output_directory,"model_config.yaml"),file_name="model_config.yaml")
    i = 0
    netpostfix = "epoch_%d_params.pt"
    optimizerpostfix = "epoch_%d_optimizer.pt"
    with experiment.train():
        while i < num_epochs:
            time.sleep(2.0)
            postfix = i + 1
            modelfile = netpostfix % (postfix-1)
            optimizerfile = optimizerpostfix % (postfix-1)
            print("Running Epoch Number %d" %(postfix))

            tick = time.time()
            run_epoch(experiment, net, optimizer, dataloader)
            tock = time.time()
            print("Finished epoch %d in %f seconds." % ( postfix , tock-tick ) )
            experiment.log_epoch_end(postfix)

            modelout = os.path.join(output_directory, modelfile)
            with open(modelout,'wb') as f:
                torch.save(net.state_dict(), f)
            experiment.log_model("epoch_%d" % (postfix,), modelout)  

            optimizerout = os.path.join(output_directory, optimizerfile)
            with open(optimizerout,'wb') as f:
                torch.save(optimizer.state_dict(), f)
            experiment.log_asset(optimizerout, optimizerfile)
            i = i + 1
import logging
if __name__ == '__main__':
    logging.basicConfig()
    go()    
    