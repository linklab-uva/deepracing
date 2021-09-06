import comet_ml
import torch
import torch.nn as NN
import torch.nn.functional as F
import torch.utils.data as data_utils
import deepracing_models.data_loading.file_datasets as FD
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
from deepracing.raceline_utils import loadBoundary
from deepracing import searchForFile
import deepracing.path_utils.geometric as geometric
import tempfile

#torch.backends.cudnn.enabled = False
def run_epoch(experiment, network, optimizer, dataloader, config, loss_func, use_tqdm = False, debug=False, plot=False):

    if use_tqdm:
        t = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        t = enumerate(dataloader)
    network.train()  # This is important to call before training!
   
    dev = next(network.parameters()).device  # we are only doing single-device training for now, so this works fine.
    dtype = next(network.parameters()).dtype # we are only doing single-device training for now, so this works fine.
    fix_first_point = config["fix_first_point"]
    bezier_order = network.params_per_dimension-1+int(fix_first_point)

    for (i, imagedict) in t:
        times = imagedict[""]
        input_images = imagedict["images"].type(dtype).to(device=dev)
        raceline_positions = (imagedict["raceline_positions"])[:,:,[0,2]].type(dtype).to(device=dev)
        batch_size = input_images.shape[0]
        
        network_output = network(input_images)
        if fix_first_point:
            initial_zeros = torch.zeros(batch_size,1,2,dtype=dtype,device=dev)
            network_output_reshape = network_output.transpose(1,2)
            predictions = torch.cat((initial_zeros,network_output_reshape),dim=1)
        else:
            predictions = network_output.transpose(1,2)

        dt = times[:,-1]-times[:,0]
        s = (times - times[:,0,None])/dt[:,None]

        Mpos, controlpoints_fit = deepracing_models.math_utils.bezier.bezierLsqfit(raceline_positions, bezier_order, t=s)
        pred_points = torch.matmul(Mpos, predictions)

        loss = loss_func(pred_points, raceline_positions)

        if debug:
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
            images_np = np.round(255.0*input_images[0].detach().cpu().numpy().copy().transpose(0,2,3,1)).astype(np.uint8)
            
            ims = []
            for i in range(images_np.shape[0]):
                ims.append([ax1.imshow(images_np[i])])
            ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True, repeat=True)
            fit_points = torch.matmul(Mpos, controlpoints_fit)

        optimizer.zero_grad()
        loss.backward() 
        # Weight and bias updates.
        optimizer.step()
        if use_tqdm:
            t.set_postfix({"current_loss" : loss.item()})
def go():
    parser = argparse.ArgumentParser(description="Train AdmiralNet Pose Predictor")
    parser.add_argument("model_config_file", type=str,  help="Model Configuration file to load")
    parser.add_argument("dataset_config_file", type=str,  help="Dataset Configuration file to load")
    parser.add_argument("output_directory", type=str, help="Where to save models.")
    parser.add_argument("--debug", action="store_true",  help="Don't actually push to comet, just testing")
    parser.add_argument("--tqdm", action="store_true",  help="Display tqdm progress bar on each epoch")
    parser.add_argument("--gpu", type=int, default=None,  help="Override the GPU index specified in the config file")

    
    args = parser.parse_args()

    config_file = args.model_config_file
    dataset_config_file = args.dataset_config_file
    main_dir = args.output_directory
    debug = args.debug
    use_tqdm = args.tqdm

    with open(dataset_config_file) as f:
        dataset_config = yaml.load(f, Loader = yaml.SafeLoader)
        
    with open(config_file) as f:
        config = yaml.load(f, Loader = yaml.SafeLoader)

    context_length = config["context_length"]
    bezier_order = config["bezier_order"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    momentum = config["momentum"]
    dampening = config["dampening"]
    nesterov = config["nesterov"]
    project_name = config["project_name"]
    fix_first_point = config["fix_first_point"]
    lookahead_time = config["lookahead_time"]
   
    if args.gpu is not None:
        gpu = args.gpu
        config["gpu"]  = gpu
    else:
        gpu = config["gpu"] 
    torch.cuda.set_device(gpu)

    num_epochs = config["num_epochs"]
    num_workers = config["num_workers"]
    hidden_dim = config["hidden_dimension"]
    use_3dconv = config["use_3dconv"]
    num_recurrent_layers = config.get("num_recurrent_layers",1)
    config["hostname"] = socket.gethostname()

    
    print("Using config:\n%s" % (str(config)))
    net = deepracing_models.nn_models.Models.AdmiralNetCurvePredictor( input_channels=3, context_length = context_length , hidden_dim = hidden_dim, num_recurrent_layers=num_recurrent_layers, params_per_dimension=bezier_order + 1 - int(fix_first_point), use_3dconv = use_3dconv) 
    print( "net:\n%s" % (str(net),) )
    loss_func : loss_functions.SquaredLpNormLoss = loss_functions.SquaredLpNormLoss()
    use_float = config["use_float"]
    if use_float:
        net = net.float()
    else:
        net = net.double()
    dtype = next(net.parameters()).dtype
    loss_func = loss_func.type(dtype)

    

        
    if gpu>=0:
        print("moving stuff to GPU")
        net = net.cuda(gpu)
        loss_func = loss_func.cuda(gpu)
    optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum = momentum, dampening=dampening, nesterov=nesterov)



    dsets=[]
    raceline_file = dataset_config["raceline_file"]
    dset_tags = dataset_config["tags"]
    for dataset in dataset_config["datasets"]:
        current_dset = FD.LocalRacelineDataset(dataset["root_folder"], raceline_file, context_length=context_length, lookahead_time=lookahead_time)
        dsets.append(current_dset)

    if len(dsets)==1:
        dset = dsets[0]
    else:
        dset = torch.utils.data.ConcatDataset(dsets)
    
    dataloader = data_utils.DataLoader(dset, batch_size=batch_size, shuffle=True, pin_memory=(gpu>=0))
    print("Dataloader of of length %d" %(len(dataloader)))
    if debug:
        print("Using datasets:\n%s", (str(dataset_config)))
    
    if debug:
        output_directory = os.path.join(main_dir, "debug")
        os.makedirs(output_directory, exist_ok=True)
        experiment = None
    else:
        experiment = comet_ml.Experiment(workspace="electric-turtle", project_name=project_name)
        output_directory = os.path.join(main_dir, experiment.get_key())
        if os.path.isdir(output_directory) :
            raise FileExistsError("%s already exists, this should not happen." %(output_directory) )
        os.makedirs(output_directory)
        experiment.log_parameters(config)
        experiment.log_parameters(dataset_config)
        if len(dset_tags)>0:
            experiment.add_tags(dset_tags)
        experiment_config = {"experiment_key": experiment.get_key()}
        yaml.dump(experiment_config, stream=open(os.path.join(output_directory,"experiment_config.yaml"),"w"), Dumper=yaml.SafeDumper)
        yaml.dump(dataset_config, stream=open(os.path.join(output_directory,"dataset_config.yaml"), "w"), Dumper = yaml.SafeDumper)
        yaml.dump(config, stream=open(os.path.join(output_directory,"model_config.yaml"), "w"), Dumper = yaml.SafeDumper)
        experiment.log_asset(os.path.join(output_directory,"dataset_config.yaml"),file_name="datasets.yaml")
        experiment.log_asset(os.path.join(output_directory,"experiment_config.yaml"),file_name="experiment_config.yaml")
        experiment.log_asset(os.path.join(output_directory,"model_config.yaml"),file_name="model_config.yaml")
        i = 0
# def run_epoch(experiment, network, optimizer, dataloader, config, use_tqdm = False, debug=False, plot=False):
    if debug:
        run_epoch(experiment, net, optimizer, dataloader, config, loss_func, debug=True, use_tqdm=use_tqdm)
    else:
        netpostfix = "epoch_%d_params.pt"
        optimizerpostfix = "epoch_%d_optimizer.pt"
        with experiment.train():
            for i in range(num_epochs):
                time.sleep(2.0)
                postfix = i + 1
                modelfile = "params.pt"
                optimizerfile = "optimizer.pt"
                run_epoch(experiment, net, optimizer, dataloader, config, loss_func, use_tqdm=use_tqdm)

                modelout = os.path.join(output_directory,modelfile)
                with open(modelout,'wb') as f:
                    torch.save(net.state_dict(), f)
                optimizerout = os.path.join(output_directory, optimizerfile)
                with open(optimizerout,'wb') as f:
                    torch.save(optimizer.state_dict(), f)
                time.sleep(1.0)

                with open(modelout,'rb') as f:
                    experiment.log_asset( f, file_name=netpostfix %(postfix,) )
                with open(optimizerout,'rb') as f:
                    experiment.log_asset( f, file_name=optimizerpostfix %(postfix,) )

import logging
if __name__ == '__main__':
    logging.basicConfig()
    go()    
    