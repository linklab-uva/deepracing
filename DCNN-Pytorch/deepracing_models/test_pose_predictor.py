import torch
import torch.utils.data as data_utils
import data_loading.proto_datasets
from tqdm import tqdm as tqdm
import nn_models.LossFunctions as loss_functions
import nn_models.Models as models
import numpy as np
import torch.optim as optim
from tqdm import tqdm as tqdm
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
import deepracing.backend
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
loss = torch.zeros(1)
def run_epoch(network, optimizer, trainLoader, gpu, position_loss, rotation_loss, loss_weights=[1.0, 1.0], imsize=(66,200), debug=False, use_tqdm=True, use_float=True, use_optflow=True):
    global loss
    cum_loss = 0.0
    cum_rotation_loss = 0.0
    cum_position_loss = 0.0
    batch_size = trainLoader.batch_size
    num_samples=0.0
    if use_tqdm:
        t = tqdm(enumerate(trainLoader), total=len(trainLoader))
    else:
        t = enumerate(trainLoader)
    network.eval() 
    for (i, (image_torch, opt_flow_torch, position_torch, rotation_torch, linear_velocity_torch, angular_velocity_torch, session_time) ) in t:
        if debug:
            images_np = image_torch[0].numpy().copy()
            num_images = images_np.shape[0]
            print(num_images)
            images_np_transpose = np.zeros((num_images, images_np.shape[2], images_np.shape[3], images_np.shape[1]), dtype=np.uint8)
            ims = []
            for i in range(num_images):
                images_np_transpose[i]=skimage.util.img_as_ubyte(images_np[i].transpose(1,2,0))
                im = plt.imshow(images_np_transpose[i], animated=True)
                ims.append([im])
            ani = animation.ArtistAnimation(plt.figure(), ims, interval=50, blit=True, repeat_delay=0)
            plt.show()
            print(position_torch)
            print(rotation_torch)
        if use_optflow:
            image_torch = torch.cat((image_torch,opt_flow_torch),axis=2)
        if use_float:
            image_torch = image_torch.float()
            position_torch = position_torch.float()
            rotation_torch = rotation_torch.float()
        else:
            image_torch = image_torch.double()
            position_torch = position_torch.double()
            rotation_torch = rotation_torch.double()

        if gpu>=0:
            image_torch = image_torch.cuda(gpu)
            position_torch = position_torch.cuda(gpu)
            rotation_torch = rotation_torch.cuda(gpu)
        images_nan = torch.sum(image_torch!=image_torch)!=0
        positions_labels_nan = torch.sum(position_torch!=position_torch)!=0
        rotation_labels_nan = torch.sum(rotation_torch!=rotation_torch)!=0
        if(images_nan):
            print(images_nan)
            raise ValueError("Input image block has a NaN!!!")
        if(rotation_labels_nan):
            print(rotation_torch)
            raise ValueError("Rotation label has a NaN!!!")
        if(positions_labels_nan):
            print(position_torch)
            raise ValueError("Position label has a NaN!!!")
      #  print(image_torch.dtype)
        # Forward pass:
        position_predictions, rotation_predictions = network(image_torch)
        positions_nan = torch.sum(position_predictions!=position_predictions)!=0
        rotation_nan = torch.sum(rotation_predictions!=rotation_predictions)!=0
        if(positions_nan):
            print(position_predictions)
            raise ValueError("Position prediction has a NaN!!!")
        if(rotation_nan):
            print(rotation_predictions)
            raise ValueError("Rotation prediction has a NaN!!!")
        #print("Output shape: ", outputs.shape)
        #print("Label shape: ", labels.shape)
        rotation_loss_ = rotation_loss(rotation_predictions, rotation_torch)
        rotation_loss_nan = torch.sum(rotation_loss_!=rotation_loss_)!=0
        if(rotation_loss_nan):
            print(rotation_loss_)
            raise ValueError("rotation_loss has a NaN!!!")
        position_loss_ = position_loss(position_predictions, position_torch)
        position_loss_nan = torch.sum(position_loss_!=position_loss_)!=0
        if(position_loss_nan):
            print(position_loss_)
            raise ValueError("position_loss has a NaN!!!")
        loss = loss_weights[0]*position_loss_ + loss_weights[1]*rotation_loss_
        loss_nan = torch.sum(loss!=loss)!=0
        if(positions_nan):
            print(loss)
            raise ValueError("loss has a NaN!!!")


        # Backward pass:
        optimizer.zero_grad()
        loss.backward() 

        # Weight and bias updates.
        optimizer.step()

        if use_tqdm:
            # logging information
            cum_loss += float(loss.item())
            cum_position_loss += float(position_loss_.item())
            cum_rotation_loss += float(rotation_loss_.item())
            num_samples += float(batch_size)
            t.set_postfix({"cum_loss" : cum_loss/num_samples, "position_loss" : cum_position_loss/num_samples, "rotation_loss" : cum_rotation_loss/num_samples})
def go():
    parser = argparse.ArgumentParser(description="Train AdmiralNet Pose Predictor")
    parser.add_argument("config_file", type=str,  help="Configuration file to load")
    parser.add_argument("model_file", type=str,  help="Model parameter file to load")
    parser.add_argument("--debug", action="store_true",  help="Display images upon each iteration of the training loop")
    args = parser.parse_args()
    config_file = args.config_file
    model_file = args.model_file
    debug = args.debug
    with open(config_file) as f:
        config = yaml.load(f, Loader = yaml.SafeLoader)

    image_size = config["image_size"]
    hidden_dimension = config["hidden_dimension"]
    input_channels = config["input_channels"]
    sequence_length = config["sequence_length"]
    context_length = config["context_length"]
    gpu = config["gpu"]
    temporal_conv_feature_factor = config["temporal_conv_feature_factor"]
    debug = config["debug"]
    use_float = config["use_float"]
    
    
    net = models.AdmiralNetPosePredictor(gpu=gpu,context_length = context_length, sequence_length = sequence_length,\
        hidden_dim=hidden_dimension, input_channels=input_channels, temporal_conv_feature_factor = temporal_conv_feature_factor)
    print(net.rnn_init_hidden)
    net.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
    print(net.rnn_init_hidden)
    if use_float:
        net = net.float()
    else:
        net = net.double()
    if gpu>=0:
        net = net.cuda(gpu)
    #print(net)
    # max_spare_txns = 1

    # #image_wrapper = deepracing.backend.ImageFolderWrapper(os.path.dirname(image_db))
    # datasets = config["datasets"]
    # dsets=[]
    # use_optflow=True
    # for dataset in datasets:
    #     print("Parsing database config: %s" %(str(dataset)))
    #     image_db = dataset["image_db"]
    #     opt_flow_db = dataset.get("opt_flow_db", "")
    #     label_db = dataset["label_db"]
    #     key_file = dataset["key_file"]
    #     label_wrapper = deepracing.backend.PoseSequenceLabelLMDBWrapper()
    #     label_wrapper.readDatabase(label_db, max_spare_txns=max_spare_txns )
    #     image_size = np.array(image_size)
    #     image_mapsize = float(np.prod(image_size)*3+12)*float(len(label_wrapper.getKeys()))*1.1
    #     image_wrapper = deepracing.backend.ImageLMDBWrapper(direct_caching=False)
    #     image_wrapper.readDatabase(image_db, max_spare_txns=max_spare_txns, mapsize=image_mapsize )
    #     optical_flow_db_wrapper = None
    #     if not opt_flow_db=='':
    #         print("Using optical flow database at %s" %(opt_flow_db))
    #         optical_flow_db_wrapper = deepracing.backend.OpticalFlowLMDBWrapper()
    #         optical_flow_db_wrapper.readDatabase(opt_flow_db, max_spare_txns=max_spare_txns, mapsize=int(round( float(image_mapsize)*8/3) ) )
    #     else:
    #         use_optflow=False
    #     curent_dset = data_loading.proto_datasets.PoseSequenceDataset(image_wrapper, label_wrapper, key_file, context_length, sequence_length,\
    #                  image_size = image_size, optical_flow_db_wrapper=optical_flow_db_wrapper)
    #     dsets.append(curent_dset)
    #     print("\n")
    # if len(dsets)==1:
    #     dset = dsets[0]
    # else:
    #     dset = torch.utils.data.ConcatDataset(dsets)
    
    # dataloader = data_utils.DataLoader(dset, batch_size=batch_size,
    #                     shuffle=True, num_workers=num_workers)
    # print("Dataloader of of length %d" %(len(dataloader)))
    # run_epoch(net, optimizer, dataloader, gpu, position_loss, rotation_loss, loss_weights=loss_weights, debug=debug, use_tqdm=True, use_float = use_float,  use_optflow = use_optflow)
     
import logging
if __name__ == '__main__':
    logging.basicConfig()
    go()    
    