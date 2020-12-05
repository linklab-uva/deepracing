import comet_ml
import torch
import torch.nn as NN
import torch.nn.functional as F
import torch.utils.data as data_utils
import deepracing_models.data_loading.proto_datasets as PD
from tqdm import tqdm as tqdm
import deepracing_models.nn_models.LossFunctions as loss_functions
import deepracing_models.nn_models.Models
import deepracing_models.nn_models.VariationalModels
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
import deepracing_models.math_utils as mu

#torch.backends.cudnn.enabled = False
def run_epoch(experiment, encoder, optimizer, dataloader, recon_loss, loss_weights, use_tqdm = False, plot=False):
    cum_loss = 0.0
    cum_param_loss = 0.0
    cum_position_loss = 0.0
    cum_velocity_loss = 0.0
    num_samples=0.0
    if use_tqdm:
        t = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        t = enumerate(dataloader)
    encoder.train()  
    dataloaderlen = len(dataloader)
    
    dev = next(encoder.parameters()).device  # we are only doing single-device training for now, so this works fine.
    dtype = next(encoder.parameters()).dtype # we are only doing single-device training for now, so this works fine.


    for (i, imagedict) in t:
        images = imagedict["image"].type(dtype).to(device=dev)
        batch_size = images.shape[0]
        
        z, recon = encoder(images)
        loss  = recon_loss(recon, images)

        
        # if plot:
        #     print("Session times: ")
        #     print(session_times)
        #     print("Normalized session times: ")
        #     print(s)
        #     fig, (axin, axrecon) = plt.subplots(nrows=1, ncols=2)
        #     images_np = np.round(255.0*images[0].detach().cpu().numpy().copy().transpose(0,2,3,1)).astype(np.uint8)
        #     reconimages_np = np.round(255.0*reconstructed_images[0].detach().cpu().numpy().copy().transpose(0,2,3,1)).astype(np.uint8)
        #     #image_np_transpose=skimage.util.img_as_ubyte(images_np[-1].transpose(1,2,0))
        #     # oap = other_agent_positions[other_agent_positions==other_agent_positions].view(1,-1,60,2)
        #     # print(oap)
        #     ims = []
        #     for i in range(images_np.shape[0]):
        #         ims.append([axin.imshow(images_np[i]), axrecon.imshow(reconimages_np[i])])
        #     ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True, repeat=True)

        #     plt.show()

        optimizer.zero_grad()
        loss.backward(retain_graph=False) 
        # Weight and bias updates.
        optimizer.step()
        # logging information
        # current_position_loss_float = float(current_position_loss.item())
        # num_samples += 1.0
        # if not debug:
        #     experiment.log_metric("current_position_loss", current_position_loss_float)
        if use_tqdm:
            t.set_postfix({"recon" : loss.item()})#, "KLD" : KLD.item()})
def go():
    parser = argparse.ArgumentParser(description="Train Image Curve Encoder")
    parser.add_argument("dataset_config_file", type=str,  help="Dataset Configuration file to load")
    parser.add_argument("model_config_file", type=str,  help="Model Configuration file to load")
    parser.add_argument("output_directory", type=str,  help="Where to put the resulting model files")

    parser.add_argument("--debug", action="store_true",  help="Don't actually push to comet, just testing")
    parser.add_argument("--plot", action="store_true",  help="Plot images upon each iteration of the training loop")
    parser.add_argument("--models_to_disk", action="store_true",  help="Save the model files to disk in addition to comet.ml")
    parser.add_argument("--tqdm", action="store_true",  help="Display tqdm progress bar on each epoch")
    parser.add_argument("--gpu", type=int, default=None,  help="Override the GPU index specified in the config file")

    
    args = parser.parse_args()

    dataset_config_file = args.dataset_config_file
    debug = args.debug
    plot = args.plot
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
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    momentum = config["momentum"]
    project_name = config["project_name"]
    manifold_dim = config["manifold_dim"]
    loss_weights = config["loss_weights"]
    use_float = config["use_float"]
   
    if args.gpu is not None:
        gpu = args.gpu
        config["gpu"]  = gpu
    else:
        gpu = config["gpu"] 

    num_epochs = config["num_epochs"]
    num_workers = config["num_workers"]

    print("Using config:\n%s" % (str(config)))
    encoder = deepracing_models.nn_models.VariationalModels.ConvolutionalAutoencoder(manifold_dim, input_channels)
  #  print("encoder:\n%s" % (str(encoder)))
     
    if use_float:
        encoder = encoder.float()
    else:
        encoder = encoder.double()
    dtype = next(encoder.parameters()).dtype
    #recon_loss = NN.BCELoss().type(dtype)
    recon_loss = NN.MSELoss().type(dtype)

    dsets=[]
    alltags = set(dataset_config.get("tags",[]))
    
    if gpu>=0:
        print("moving stuff to GPU")
        device = torch.device("cuda:%d" % gpu)
        encoder = encoder.cuda(gpu)
        recon_loss = recon_loss.cuda(gpu)
    else:
        device = torch.device("cpu")
    optimizer = optim.SGD(encoder.parameters(), lr = learning_rate, momentum=momentum)



    image_size = dataset_config["image_size"]
    for dataset in dataset_config["datasets"]:
        dlocal : dict = {k: dataset_config[k] for k in dataset_config.keys()  if (not (k in ["datasets"]))}
        dlocal.update(dataset)
        print("Parsing database config: %s" %(str(dlocal)))
        root_folder = dlocal["root_folder"]
        dataset_tags = dlocal.get("tags", [])
        alltags = alltags.union(set(dataset_tags))
        image_folder = os.path.join(root_folder,"images")

        image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder,f)) and (os.path.splitext(f)[-1].lower() in {".jpg", ".png"})]
        image_mapsize = int(float(np.prod(image_size)*3+12)*float(len(image_files))*1.1)

        image_lmdb_folder = os.path.join(image_folder,"image_lmdb")
        image_wrapper = deepracing.backend.ImageLMDBWrapper()
        image_wrapper.readDatabase( image_lmdb_folder , mapsize=image_mapsize )
        keys = image_wrapper.getKeys()
        current_dset = PD.ImageDataset(image_wrapper, keys=keys, image_size=image_size)
        dsets.append(current_dset)
        print("\n")
    if len(dsets)==1:
        dset = dsets[0]
    else:
        dset = torch.utils.data.ConcatDataset(dsets)
    
    dataloader = data_utils.DataLoader(dset, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers, pin_memory=gpu>=0)
    print("Dataloader of of length %d" %(len(dataloader)))
    if debug:
        print("Using datasets:\n%s", (str(dataset_config)))
    
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
    if debug:
        for asdf in range(1,10):
            run_epoch(experiment, encoder, optimizer, dataloader, recon_loss, loss_weights, use_tqdm=True, plot=plot)
    else:
        encoderpostfix = "epoch_%d_encoder.pt"
        decoderpostfix = "epoch_%d_decoder.pt"
        optimizerpostfix = "epoch_%d_optimizer.pt"
        with experiment.train():
            while i < num_epochs:
                time.sleep(2.0)
                postfix = i + 1
                if models_to_disk:
                    encoderfile = encoderpostfix % (postfix-1)
                    optimizerfile = optimizerpostfix % (postfix-1)
                else:
                    encoderfile = "encoder.pt"
                    optimizerfile = "optimizer.pt"
                print("Running Epoch Number %d" %(postfix))
                #dset.clearReaders()
                tick = time.time()
                run_epoch(experiment, encoder, optimizer, dataloader, recon_loss, loss_weights, use_tqdm=use_tqdm)
                tock = time.time()
                print("Finished epoch %d in %f seconds." % ( postfix , tock-tick ) )
                experiment.log_epoch_end(postfix)
                

                encoderout = os.path.join(output_directory,encoderfile)
                with open(encoderout,'wb') as f:
                    torch.save(encoder.state_dict(), f)
                with open(encoderout,'rb') as f:
                    experiment.log_asset(f,file_name=encoderpostfix %(postfix,) )

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
    