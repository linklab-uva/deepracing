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
def run_epoch(network, optimizer, trainLoader, gpu, kinematic_loss, taylor_loss, loss_weights=[1.0, 1.0, 1.0, 1.0], imsize=(66,200), debug=False, use_tqdm=True, use_float=True):
    cum_loss = 0.0
    cum_position_loss = 0.0
    cum_linear_loss = 0.0
    cum_angular_loss = 0.0
    cum_taylor_loss_1 = 0.0
    cum_taylor_loss_2 = 0.0
    batch_size = trainLoader.batch_size
    num_samples=0.0
    loss = torch.zeros(1)
    if use_tqdm:
        t = tqdm(enumerate(trainLoader), total=len(trainLoader))
    else:
        t = enumerate(trainLoader)
    network.train()  # This is important to call before training!
    for (i, (image_torch, opt_flow_torch, position_torch, _, linear_velocity_torch, _, session_time_torch) ) in t:
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
        velocity_loss = kinematic_loss(velocity_predictions, speeds)
       # taylor_loss_1 = taylor_loss(position_predictions, linear_velocity_torch, session_time_torch)
        #taylor_loss_2 = taylor_loss(position_torch, velocity_predictions, session_time_torch)

        loss = loss_weights[0]*position_loss + loss_weights[1]*velocity_loss# + \
           # loss_weights[2]*taylor_loss_1 + loss_weights[3]*taylor_loss_2
        loss_nan = torch.sum(loss!=loss)!=0
        if(loss_nan):
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
            cum_position_loss += float(position_loss.item())
            cum_linear_loss += float(velocity_loss.item())
          #  cum_taylor_loss_1 += float(taylor_loss_1.item())
            #cum_taylor_loss_2 += float(taylor_loss_2.item())
            num_samples += float(batch_size)
            t.set_postfix({"cum_loss" : cum_loss/num_samples, "position_loss" : cum_position_loss/num_samples, "velocity_loss" : cum_linear_loss/num_samples})#, \
                         #   "taylor_loss_1" : cum_taylor_loss_1/num_samples, "taylor_loss_2" : cum_taylor_loss_2/num_samples})
def go():
    parser = argparse.ArgumentParser(description="Train AdmiralNet Pose Predictor")
    parser.add_argument("config_file", type=str,  help="Configuration file to load")
    parser.add_argument("output_directory", type=str,  help="Where to put the resulting model files")

    parser.add_argument("--epochstart", type=int, default=1,  help="Restart training from the given epoch number")
    parser.add_argument("--debug", action="store_true",  help="Display images upon each iteration of the training loop")
    parser.add_argument("--override", action="store_true",  help="Delete output directory and replace with new data")
    parser.add_argument("--tqdm", action="store_true",  help="Display tqdm progress bar on each epoch")
    parser.add_argument("--gpu", type=int, default=None,  help="Override the GPU index specified in the config file")
    args = parser.parse_args()
    config_file = args.config_file
    output_directory = args.output_directory
    debug = args.debug
    epochstart = args.epochstart
    with open(config_file) as f:
        config = yaml.load(f, Loader = yaml.SafeLoader)

    image_size = config["image_size"]
    hidden_dimension = config["hidden_dimension"]
    input_channels = config["input_channels"]
    sequence_length = config["sequence_length"]
    context_length = config["context_length"]
    if args.gpu is not None:
        gpu = args.gpu
    else:
        gpu = config["gpu"] 
    loss_weights = config["loss_weights"]
    temporal_conv_feature_factor = config["temporal_conv_feature_factor"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    momentum = config["momentum"]
    num_epochs = config["num_epochs"]
    num_workers = config["num_workers"]
    debug = config["debug"]
    position_loss_reduction = config["position_loss_reduction"]
    use_float = config["use_float"]
    learnable_initial_state = config.get("learnable_initial_state",True)
    downsample_method=config["downsample_method"]
    print("Using config:\n%s" % (str(config)))
    net = nn_models.Models.AdmiralNetKinematicPredictor(context_length = context_length, sequence_length = sequence_length,\
        hidden_dim=hidden_dimension, output_dimension=4, input_channels=input_channels, learnable_initial_state = learnable_initial_state) 
    print("net:\n%s" % (str(net)))

    kinematics_loss = torch.nn.MSELoss(reduction=position_loss_reduction)
    taylor_loss = loss_functions.TaylorSeriesLinear(reduction=position_loss_reduction)
    losses = NN.ModuleList([kinematics_loss,taylor_loss])
    if use_float:
        print("casting stuff to float")
        net = net.float()
        losses = losses.float()
    else:
        print("casting stuff to double")
        net = net.double()
        losses = losses.double()
    if gpu>=0:
        print("moving stuff to GPU")
        net = net.cuda(gpu)
        losses = losses.cuda(gpu)
    print("losses:\n%s" % (str(losses)))
    optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum = momentum, dampening=0.000, nesterov=True)
    netpostfix = "epoch_%d_params.pt"
    optimizerpostfix = "epoch_%d_optimizer.pt"
    if epochstart>1:
        net.load_state_dict(torch.load(os.path.join(output_directory,netpostfix %(epochstart)), map_location=next(net.parameters()).device))
        optimizer.load_state_dict(torch.load(os.path.join(output_directory,optimizerpostfix %(epochstart)), map_location=next(net.parameters()).device))
    else:
        if (not args.override) and os.path.isdir(output_directory) :
            s = ""
            while(not (s=="y" or s=="n")):
                s = input("Directory " + output_directory + " already exists. Overwrite it with new data? [y\\n]\n")
            if s=="n":
                print("Thanks for playing!")
                exit(0)
            shutil.rmtree(output_directory)
        elif os.path.isdir(output_directory):
            shutil.rmtree(output_directory)
        os.makedirs(output_directory, exist_ok=True)
    
    if num_workers == 0:
        max_spare_txns = 50
    else:
        max_spare_txns = num_workers

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
                     image_size = image_size, optical_flow_db_wrapper=optical_flow_db_wrapper, downsample=downsample_method)
        dsets.append(curent_dset)
        print("\n")
    if len(dsets)==1:
        dset = dsets[0]
    else:
        dset = torch.utils.data.ConcatDataset(dsets)
    
    dataloader = data_utils.DataLoader(dset, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)
    print("Dataloader of of length %d" %(len(dataloader)))
    yaml.dump(config, stream=open(os.path.join(output_directory,"config.yaml"), "w"), Dumper = yaml.SafeDumper)
    if(epochstart==1):
        i = 0
    else:
        i = epochstart
    while i < num_epochs:
        time.sleep(2.0)
        postfix = i + 1
        print("Running Epoch Number %d" %(postfix))
        #dset.clearReaders()
        try:
            tick = time.time()
            run_epoch(net, optimizer, dataloader, gpu, kinematics_loss, taylor_loss, loss_weights=loss_weights, debug=debug, use_tqdm=args.tqdm, use_float = use_float)
            tock = time.time()
            print("Finished epoch %d in %f seconds." % ( postfix , tock-tick ) )
        except Exception as e:
            print("Restarting epoch %d because %s"%(postfix, str(e)))
            modelin = os.path.join(output_directory, netpostfix %(postfix-1))
            optimizerin = os.path.join(output_directory,optimizerpostfix %(postfix-1))
            net.load_state_dict(torch.load(modelin))
            optimizer.load_state_dict(torch.load(optimizerin))
            continue
        modelout = os.path.join(output_directory,netpostfix %(postfix))
        torch.save(net.state_dict(), modelout)
        
        optimizerout = os.path.join(output_directory,optimizerpostfix %(postfix))
        torch.save(optimizer.state_dict(), optimizerout)
        
        i = i + 1
import logging
if __name__ == '__main__':
    logging.basicConfig()
    go()    
    