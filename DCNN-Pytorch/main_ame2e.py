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
def run_epoch(experiment, network, optimizer, dataloader, control_loss, use_tqdm = False, debug=False, plot=False):
    cum_loss = 0.0
    cum_param_loss = 0.0
    cum_position_loss = 0.0
    cum_velocity_loss = 0.0
    num_samples=0.0
    if use_tqdm:
        t = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        t = enumerate(dataloader)
    network.train()  # This is important to call before training!
    dataloaderlen = len(dataloader)
    firstparam = next(network.parameters()) # we are only doing single-device training for now, so this works fine.
    dev = firstparam.device  
    dtype = firstparam.dtype
    for (i, imagedict) in t:
        input_images = imagedict["images"].type(dtype).to(device=dev)
        steering = imagedict["steering"]
        throttle = imagedict["throttle"]
        brake = imagedict["brake"]
        targets = torch.stack([steering, throttle-brake], dim=2).type(dtype).to(device=dev)

        predictions = network(input_images)


        loss = control_loss(predictions, targets)
        optimizer.zero_grad()
        loss.backward() 
        # Weight and bias updates.
        optimizer.step()
        # logging information
        current_control_loss = float(loss.item())
        num_samples += 1.0
        if not debug:
            experiment.log_metric("current_control_loss", current_control_loss)
        if use_tqdm:
            t.set_postfix({"current_control_loss" : current_control_loss})
def go():
    parser = argparse.ArgumentParser(description="Train AdmiralNet Waypoint Predictor")
    parser.add_argument("dataset_config_file", type=str,  help="Dataset Configuration file to load")
    parser.add_argument("model_config_file", type=str,  help="Model Configuration file to load")
    parser.add_argument("output_directory", type=str,  help="Where to put the resulting model files")

    parser.add_argument("--debug", action="store_true",  help="Don't actually push to comet, just testing")
    parser.add_argument("--plot", action="store_true",  help="Plot images upon each iteration of the training loop")
    parser.add_argument("--model_load",  type=str, default=None,  help="Load this model file prior to running. usually in conjunction with debug")
    parser.add_argument("--models_to_disk", action="store_true",  help="Save the model files to disk in addition to comet.ml")
    parser.add_argument("--tqdm", action="store_true",  help="Display tqdm progress bar on each epoch")
    parser.add_argument("--gpu", type=int, default=None,  help="Override the GPU index specified in the config file")

    
    args = parser.parse_args()

    dataset_config_file = args.dataset_config_file
    debug = args.debug
    plot = args.plot
    model_load = args.model_load
    models_to_disk = args.models_to_disk
    use_tqdm = args.tqdm

    with open(dataset_config_file) as f:
        dataset_config = yaml.load(f, Loader = yaml.SafeLoader)
    config_file = args.model_config_file
    with open(config_file) as f:
        config = yaml.load(f, Loader = yaml.SafeLoader)
    print(dataset_config)
    image_size = dataset_config["image_size"]
    input_channels = config["input_channels"]
    
    context_length = config["context_length"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    momentum = config["momentum"]
    dampening = config["dampening"]
    project_name = config["project_name"]
   
    if args.gpu is not None:
        gpu = args.gpu
        config["gpu"]  = gpu
    else:
        gpu = config["gpu"] 
    torch.cuda.set_device(gpu)

    num_epochs = config["num_epochs"]
    num_workers = config["num_workers"]
    hidden_dim = config["hidden_dimension"]
    sequence_length = config["sequence_length"]
    use_3dconv = config["use_3dconv"]
    use_float = config["use_float"]
    num_recurrent_layers = config.get("num_recurrent_layers",1)
    config["hostname"] = socket.gethostname()

    
    print("Using config:\n%s" % (str(config)))
    net = deepracing_models.nn_models.Models.AdmiralNetKinematicPredictor( context_length = context_length , input_channels=input_channels, sequence_length=sequence_length,  hidden_dim = hidden_dim, num_recurrent_layers=num_recurrent_layers, use_3dconv = use_3dconv ) 
    print("net:\n%s" % (str(net)))
    



    

    #image_wrapper = deepracing.backend.ImageFolderWrapper(os.path.dirname(image_db))
    

    if gpu>=0:
        print("moving stuff to GPU")
        device = torch.device("cuda:%d" % gpu)
    else:
        device = torch.device("cpu")
    if use_float:
        net = net.float().to(device=device)
    else:
        net = net.double().to(device=device)
    dtype = (next(net.parameters())).dtype
    control_loss = NN.MSELoss().type(dtype).to(device=device)

    optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum = momentum, dampening=dampening)

    dsets=[]
    dsetfolders = []
    alltags = set(dataset_config.get("tags",[]))
    dset_output_lengths=[]
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

        label_lmdb_folder = os.path.join(label_folder,"lmdb")
        with open(os.path.join(label_lmdb_folder,"args.yaml"), "r") as f:
            dataset.update(yaml.load(f, Loader=yaml.SafeLoader))
        label_wrapper = deepracing.backend.ControlLabelLMDBWrapper()
        label_wrapper.readDatabase( label_lmdb_folder )


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
        
        current_dset = PD.ControlOutputSequenceDataset(image_wrapper, label_wrapper, key_file, context_length=context_length, sequence_length=sequence_length)
        dsets.append(current_dset)
        
        print("\n")
    if len(dsets)==1:
        dset = dsets[0]
    else:
        dset = torch.utils.data.ConcatDataset(dsets)
    
    dataloader = data_utils.DataLoader(dset, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers, pin_memory=gpu>=0)
    print("Dataloader of of length %d" %(len(dataloader)))

    
    main_dir = args.output_directory
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
        dsetsjson = json.dumps(dataset_config, indent=1)
        experiment.log_parameter("datasets",dsetsjson)
        experiment.log_text(dsetsjson)
        if len(alltags)>0:
            experiment.add_tags(list(alltags))
        experiment_config = {"experiment_key": experiment.get_key()}
        yaml.dump(experiment_config, stream=open(os.path.join(output_directory,"experiment_config.yaml"),"w"), Dumper=yaml.SafeDumper)
        yaml.dump(dataset_config, stream=open(os.path.join(output_directory,"dataset_config.yaml"), "w"), Dumper = yaml.SafeDumper)
        yaml.dump(config, stream=open(os.path.join(output_directory,"model_config.yaml"), "w"), Dumper = yaml.SafeDumper)
        experiment.log_asset(os.path.join(output_directory,"dataset_config.yaml"),file_name="datasets.yaml")
        experiment.log_asset(os.path.join(output_directory,"experiment_config.yaml"),file_name="experiment_config.yaml")
        experiment.log_asset(os.path.join(output_directory,"model_config.yaml"),file_name="model_config.yaml")
        i = 0
        #def run_epoch(experiment, net, optimizer, dataloader, raceline_loss, other_agent_loss, config)
    if debug:
        run_epoch(experiment, net, optimizer, dataloader, control_loss, debug=True, use_tqdm=True, plot=plot)
    else:
        netpostfix = "epoch_%d_params.pt"
        optimizerpostfix = "epoch_%d_optimizer.pt"
        with experiment.train():
            while i < num_epochs:
                time.sleep(2.0)
                postfix = i + 1
                if models_to_disk:
                    modelfile = netpostfix % (postfix-1)
                    optimizerfile = optimizerpostfix % (postfix-1)
                else:
                    modelfile = "params.pt"
                    optimizerfile = "optimizer.pt"
                print("Running Epoch Number %d" %(postfix))
                #dset.clearReaders()
                try:
                    tick = time.time()
                    run_epoch(experiment, net, optimizer, dataloader, control_loss, use_tqdm=use_tqdm)
                    tock = time.time()
                    print("Finished epoch %d in %f seconds." % ( postfix , tock-tick ) )
                    experiment.log_epoch_end(postfix)
                except FileExistsError as e:
                    raise e
                except Exception as e:
                    print("Restarting epoch %d because %s"%(postfix, str(e)))
                    modelin = os.path.join(output_directory, modelfile)
                    optimizerin = os.path.join(output_directory,optimizerfile)
                    net.load_state_dict(torch.load(modelin))
                    optimizer.load_state_dict(torch.load(optimizerin))
                    continue

                modelout = os.path.join(output_directory,modelfile)
                with open(modelout,'wb') as f:
                    torch.save(net.state_dict(), f)
                with open(modelout,'rb') as f:
                    experiment.log_asset(f,file_name=netpostfix %(postfix,) )

                optimizerout = os.path.join(output_directory, optimizerfile)
                with open(optimizerout,'wb') as f:
                    torch.save(optimizer.state_dict(), f)
                with open(optimizerout,'rb') as f:
                    experiment.log_asset(f,file_name=optimizerpostfix %(postfix,) )
                i = i + 1
import logging
if __name__ == '__main__':
    logging.basicConfig()
    go()    
    