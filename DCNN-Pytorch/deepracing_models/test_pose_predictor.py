import torch
import torch.nn as NN
import torch.utils.data as data_utils
import data_loading.proto_datasets
from tqdm import tqdm as tqdm
import nn_models.LossFunctions as loss_functions
import nn_models.Models
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
def run_epoch(network, trainLoader, gpu, kinematic_loss, taylor_loss, loss_weights=[1.0, 1.0, 1.0, 1.0], imsize=(66,200), debug=False,  use_float=True):
    poslossfloat = 0.0
    batch_size = trainLoader.batch_size
    num_samples=0.0
    loss = torch.zeros(1)
    t : tqdm = tqdm(enumerate(trainLoader), total=len(trainLoader))
    network.eval()  # This is important to call before training!
    for (i, (image_torch, opt_flow_torch, position_torch, _, linear_velocity_torch, _, session_time_torch) ) in t:
        if network.input_channels==5:
            image_torch = torch.cat((image_torch,opt_flow_torch),axis=2)
        if use_float:
            image_torch = image_torch.float()
            position_torch = position_torch.float()
            linear_velocity_torch = linear_velocity_torch.float()
            session_time_torch = session_time_torch.float()
        else:
            image_torch = image_torch.double()
            position_torch = position_torch.double()
            linear_velocity_torch = linear_velocity_torch.double()
            session_time_torch = session_time_torch.double()

        if gpu>=0:
            image_torch = image_torch.cuda(gpu)
            position_torch = position_torch.cuda(gpu)
            linear_velocity_torch = linear_velocity_torch.cuda(gpu)
            session_time_torch = session_time_torch.cuda(gpu)
        images_nan = torch.sum(image_torch!=image_torch)!=0
        positions_labels_nan = torch.sum(position_torch!=position_torch)!=0
 
        predictions = network(image_torch)
        speeds = torch.norm(linear_velocity_torch,dim=2)
        position_predictions = predictions[:,:,0:3]
        velocity_predictions = predictions[:,:,3].squeeze()
        position_loss = kinematic_loss(position_predictions, position_torch)
        # velocity_loss = kinematic_loss(velocity_predictions, speeds)
        poslossfloat += position_loss.item()
        num_samples += 1.0
        #loss = loss_weights[0]*position_loss + loss_weights[1]*velocity_loss
        if debug:
            images_np = image_torch[0].cpu().numpy().copy()
            num_images = images_np.shape[0]
            print(num_images)
            images_np_transpose = np.zeros((num_images, images_np.shape[2], images_np.shape[3], images_np.shape[1]), dtype=np.uint8)
            ims = []
            for i in range(num_images):
                images_np_transpose[i]=skimage.util.img_as_ubyte(images_np[i].transpose(1,2,0))
                im = plt.imshow(images_np_transpose[i], animated=True)
                ims.append([im])
            ani = animation.ArtistAnimation(plt.figure(), ims, interval=50, blit=True, repeat_delay=0)
            gt_positions = position_torch[0].detach().cpu().numpy()
            times = np.linspace(0,1,len(gt_positions))
            pred_positions = position_predictions[0].detach().cpu().numpy()
            fig = plt.figure()
            position_ax = fig.add_subplot()
            position_ax.plot(gt_positions[:,0], gt_positions[:,2], 'ro')
            position_ax.plot(pred_positions[:,0], pred_positions[:,2], 'bo')
            plt.show()
        t.set_postfix({"posloss":poslossfloat/num_samples})
def go():
    parser = argparse.ArgumentParser(description="Train AdmiralNet Pose Predictor")
    parser.add_argument("config_file", type=str,  help="Configuration file to load")
    parser.add_argument("model_file", type=str,  help="Weight file to load")

    parser.add_argument("--gpu", type=int, default=-1,  help="GPU to use")
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
    gpu = args.gpu
    loss_weights = config["loss_weights"]
    temporal_conv_feature_factor = config["temporal_conv_feature_factor"]
    debug = args.debug
    use_float = config["use_float"]
    learnable_initial_state = config.get("learnable_initial_state",True)
    print("Using config:\n%s" % (str(config)))
    net = nn_models.Models.AdmiralNetKinematicPredictor(context_length = context_length, sequence_length = sequence_length,\
        hidden_dim=hidden_dimension, output_dimension=4, input_channels=input_channels, learnable_initial_state = learnable_initial_state) 
    net.load_state_dict(torch.load(model_file,map_location=torch.device("cpu")))
    print("net:\n%s" % (str(net)))

    kinematics_loss = torch.nn.MSELoss(reduction="mean")
    taylor_loss = loss_functions.TaylorSeriesLinear(reduction="mean")
    if use_float:
        print("casting stuff to float")
        net = net.float()
        kinematics_loss = kinematics_loss.float()
    else:
        print("casting stuff to double")
        net = net.double()
        kinematics_loss = kinematics_loss.double()
    if gpu>=0:
        print("moving stuff to GPU")
        net = net.cuda(gpu)
        kinematics_loss = kinematics_loss.cuda(gpu)
    
    
    num_workers = 0
    max_spare_txns = 50
    #image_wrapper = deepracing.backend.ImageFolderWrapper(os.path.dirname(image_db))
    datasets = config["datasets"]
    dsets=[]
    use_optflow=True
    for dataset in datasets:
        print("Parsing database config: %s" %(str(dataset)))
        image_db = dataset["image_db"]
        opt_flow_db = dataset.get("opt_flow_db", "")
        label_db = dataset["label_db"]
        key_file = dataset["key_file"]
        label_wrapper = deepracing.backend.PoseSequenceLabelLMDBWrapper()
        label_wrapper.readDatabase(label_db, max_spare_txns=max_spare_txns )
        image_size = np.array(image_size)
        image_mapsize = float(np.prod(image_size)*3+12)*float(len(label_wrapper.getKeys()))*1.1
        image_wrapper = deepracing.backend.ImageLMDBWrapper(direct_caching=False)
        image_wrapper.readDatabase(image_db, max_spare_txns=max_spare_txns, mapsize=image_mapsize )
        optical_flow_db_wrapper = None
        if not opt_flow_db=='':
            print("Using optical flow database at %s" %(opt_flow_db))
            optical_flow_db_wrapper = deepracing.backend.OpticalFlowLMDBWrapper()
            optical_flow_db_wrapper.readDatabase(opt_flow_db, max_spare_txns=max_spare_txns, mapsize=int(round( float(image_mapsize)*8/3) ) )
        else:
            use_optflow=False
        curent_dset = data_loading.proto_datasets.PoseSequenceDataset(image_wrapper, label_wrapper, key_file, context_length, sequence_length,\
                     image_size = image_size, optical_flow_db_wrapper=optical_flow_db_wrapper)
        dsets.append(curent_dset)
        print("\n")
    if len(dsets)==1:
        dset = dsets[0]
    else:
        dset = torch.utils.data.ConcatDataset(dsets)
    
    dataloader = data_utils.DataLoader(dset, batch_size=1,
                        shuffle=True, num_workers=num_workers)
    print("Dataloader of of length %d" %(len(dataloader)))
    run_epoch(net, dataloader, gpu, kinematics_loss, taylor_loss, loss_weights=loss_weights, debug=debug, use_float = use_float)
import logging
if __name__ == '__main__':
    logging.basicConfig()
    go()    
    