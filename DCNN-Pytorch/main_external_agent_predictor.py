from operator import pos
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
def run_epoch(experiment : comet_ml.Experiment, network : ExternalAgentCurvePredictor, optimizer : torch.optim.Optimizer, dataloader : data_utils.DataLoader, weighted_loss : bool = False, debug : bool = False, use_tqdm = False):
    num_samples=0.0
    if use_tqdm:
        t = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        t = enumerate(dataloader)
    network.train()  # This is important to call before training!
    
    # we are only doing single-device training for now, so this works fine.
    dev = next(network.parameters()).device
    dtype = next(network.parameters()).dtype 
    lossf=0
    for (i, datadict) in t:
        valid_mask = datadict["valid_mask"] 
        past_positions = datadict["past_positions"]
        past_velocities = datadict["past_velocities"]
        past_quaternions = datadict["past_quaternions"]
        future_positions = datadict["future_positions"]
        tfuture = datadict["tfuture"]

        valid_past_positions = (past_positions[valid_mask].type(dtype).to(dev))[:,:,[0,2]]
        valid_past_velocities = (past_velocities[valid_mask].type(dtype).to(dev))[:,:,[0,2]]
        valid_past_quaternions = past_quaternions[valid_mask].type(dtype).to(dev)
        valid_future_positions = (future_positions[valid_mask].type(dtype).to(dev))[:,:,[0,2]]
        valid_tfuture = tfuture[valid_mask].type(dtype).to(dev)
        if network.input_dim==4:
            networkinput = torch.cat([valid_past_positions, valid_past_velocities], dim=2)
        elif network.input_dim==8:
            networkinput = torch.cat([valid_past_positions, valid_past_velocities, valid_past_quaternions], dim=2)
        else:
            raise ValueError("Currently, only input dimensions of 4 and 8 are supported")
        output = network(networkinput)
        curves = torch.cat([valid_future_positions[:,0].unsqueeze(1), output], dim=1)
        if debug:
            pass

        dt = valid_tfuture[:,-1]-valid_tfuture[:,0]
        s_torch_cur = (valid_tfuture - valid_tfuture[:,0,None])/dt[:,None]
        Mpos = mu.bezierM(s_torch_cur, network.bezier_order)
        pred_points = torch.matmul(Mpos, curves)
        deltas = pred_points - valid_future_positions
        squared_norms = torch.sum(torch.square(deltas), dim=2)
        if weighted_loss:
            weights = torch.ones_like(squared_norms)
            istart = int(round(weights.shape[1]/2))
            weights[:,istart:] = torch.linspace(1.0, 0.1, steps=weights.shape[1]-istart, device=weights.device, dtype=weights.dtype)
            loss = torch.mean(weights*squared_norms)
        else:
            loss = torch.mean(squared_norms)

        optimizer.zero_grad()
        loss.backward() 
        # Weight and bias updates.
        optimizer.step()
        # logging information
        curr_loss = loss.item()
        lossf += curr_loss
        if((i%15)==0):
            experiment.log_metric("loss", curr_loss)
        if use_tqdm:
            t.set_postfix({"current_position_loss" : lossf/(i+1)})
def go():
    parser = argparse.ArgumentParser(description="Train AdmiralNet Pose Predictor")
    parser.add_argument("training_config_file", type=str,  help="Training Configuration file to load")
    parser.add_argument("model_config_file", type=str,  help="Model Configuration file to load")
    parser.add_argument("output_directory", type=str,  help="Where to put the resulting model files")

    parser.add_argument("--debug", action="store_true",  help="Don't actually push to comet, just testing")
    parser.add_argument("--gpu", type=int, default=None,  help="Override the GPU index specified in the config file")
    parser.add_argument("--clean-after-epoch", action="store_true", help="Delete model files once they get uploaded to comet")
    parser.add_argument("--tqdm", action="store_true", help="Delete model files once they get uploaded to comet")

    
    args = parser.parse_args()
    argdict = vars(args)
    print(argdict)

    training_config_file = argdict["training_config_file"]
    model_config_file = argdict["model_config_file"]
    debug = argdict["debug"]
    use_tqdm = argdict["tqdm"]

    with open(training_config_file) as f:
        training_config = yaml.load(f, Loader = yaml.SafeLoader)
    with open(model_config_file) as f:
        model_config = yaml.load(f, Loader = yaml.SafeLoader)
    
    bezier_order = model_config["bezier_order"]
    output_dim = model_config["output_dim"]
    hidden_dim = model_config["hidden_dim"]
    num_layers = model_config["num_layers"]
    dropout = model_config["dropout"]
    bidirectional = model_config["bidirectional"]
    include_rotations = model_config["include_rotations"]
    if include_rotations:
        input_dim = 8
    else:
        input_dim = 4
    model_config["input_dim"] = input_dim

    batch_size = training_config["batch_size"]
    learning_rate = training_config["learning_rate"]
    momentum = training_config["momentum"]
    nesterov = training_config["nesterov"]
    project_name = training_config["project_name"]
    num_epochs = training_config["num_epochs"]
    weighted_loss = training_config["weighted_loss"]
    prediction_time = training_config["prediction_time"]
    context_time = training_config["context_time"]
    context_indices = training_config.get("context_indices",5)
    adam = training_config.get("adam",False)
    training_config["adam"] = adam
   
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
    if adam:
        optimizer : optim.Optimizer = optim.Adam(net.parameters(), lr = learning_rate)
    else:
        optimizer : optim.Optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum=momentum, nesterov=nesterov)

    dsets=[]
    for dataset in training_config["datasets"]:
        root_folder = dataset["root_folder"]
        current_dset = PD.PoseVelocityDataset(root_folder, context_time=context_time, context_indices=context_indices, prediction_time=prediction_time)
        dsets.append(current_dset)
    if len(dsets)==1:
        dset = dsets[0]
    else:
        dset = torch.utils.data.ConcatDataset(dsets)
    
    dataloader = data_utils.DataLoader(dset, batch_size=batch_size, shuffle=True, pin_memory=(gpu>=0))

    print("Dataloader of of length %d" %(len(dataloader)))
    if debug:
        print("Using datasets:\n%s", (str(training_config["datasets"])))
    
    main_dir = args.output_directory

    experiment = comet_ml.Experiment(workspace="electric-turtle", project_name=project_name, auto_metric_logging=False)
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
    
    with experiment.train():
        for i in range(num_epochs):
            time.sleep(2.0)
            postfix = i + 1
            print("Running Epoch Number %d" %(postfix))

            tick = time.time()
            run_epoch(experiment, net, optimizer, dataloader, use_tqdm=use_tqdm, weighted_loss = weighted_loss)
            tock = time.time()
            experiment.log_epoch_end(postfix)
            print("Finished epoch %d in %f seconds." % ( postfix , tock-tick ) )

            epoch_directory = os.path.join(output_directory, "epoch_%d" % (postfix,) )
            os.makedirs(epoch_directory, exist_ok=True)

            paramsfile = os.path.join(epoch_directory, "params.pt")
            with open(paramsfile,'wb') as f:
                torch.save(net.state_dict(), f)

            optimizerfile = os.path.join(epoch_directory, "optimizer.pt")
            with open(optimizerfile,'wb') as f:
                torch.save(optimizer.state_dict(), f)

            experiment.log_asset(paramsfile, "epoch_%d_params.pt" % (postfix,), copy_to_tmp=True)
            experiment.log_asset(optimizerfile, "epoch_%d_optimizer.pt" % (postfix,),  copy_to_tmp=True)

            if argdict["clean_after_epoch"]:
                shutil.rmtree(epoch_directory)
import logging
if __name__ == '__main__':
    logging.basicConfig()
    go()    
    