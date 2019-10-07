import torch
import torch.nn as NN
import torch.utils.data as data_utils
import data_loading.proto_datasets
from tqdm import tqdm as tqdm
import nn_models.LossFunctions as params_losstions
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
import math_utils.bezier
#torch.backends.cudnn.enabled = False
def run_epoch(network, optimizer, trainLoader, gpu, kinematic_loss, loss_weights, imsize=(66,200), timewise_weights=None, debug=False, use_tqdm=True, use_float=True):
    cum_loss = 0.0
    num_samples=0.0
    batch_size = trainLoader.batch_size
    if use_tqdm:
        t = tqdm(enumerate(trainLoader), total=len(trainLoader))
    else:
        t = enumerate(trainLoader)
    network.train()
    
    for (i, (image_torch, opt_flow_torch, positions_torch, quats_torch, linear_velocities_torch, angular_velocities_torch, session_times_torch) ) in t:
        if network.input_channels==5:
            image_torch = torch.cat((image_torch,opt_flow_torch),axis=2)
        if use_float:
            image_torch = image_torch.float()
            positions_torch = positions_torch.float()
            session_times_torch = session_times_torch.float()
        else:
            image_torch = image_torch.double()
            positions_torch = positions_torch.double()
            session_times_torch = session_times_torch.double()
        if gpu>=0:
            image_torch = image_torch.cuda(gpu)
            positions_torch = positions_torch.cuda(gpu)
            session_times_torch = session_times_torch.cuda(gpu)
        #print(image_torch.shape)
        predictions = network(image_torch)
        fitpoints = positions_torch[:,:,[0,2]]
        if debug:
            images_np = image_torch[0].detach().cpu().numpy().copy()
            num_images = images_np.shape[0]
            print(num_images)
            images_np_transpose = np.zeros((num_images, images_np.shape[2], images_np.shape[3], images_np.shape[1]), dtype=np.uint8)
            ims = []
            for i in range(num_images):
                images_np_transpose[i]=skimage.util.img_as_ubyte(images_np[i].transpose(1,2,0))
                im = plt.imshow(images_np_transpose[i], animated=True)
                ims.append([im])
            ani = animation.ArtistAnimation(plt.figure(), ims, interval=250, blit=True, repeat_delay=2000)
            fig = plt.figure()
            ax = fig.add_subplot()
            fitpointsnp = fitpoints[0,:].detach().cpu().numpy().copy()
            ax.plot(fitpointsnp[:,0],fitpointsnp[:,1],'r-')
            
            #skipn = 20
            #ax.quiver(Pbeziertorch[::skipn,0].numpy(),Pbeziertorch[::skipn,1].numpy(),Pbeziertorchderiv[::skipn,0].numpy(),Pbeziertorchderiv[::skipn,1].numpy())
            #ax.plot(bezier_control_points[i,:,0].numpy(),bezier_control_points[i,:,1].numpy(),'go')
            plt.show()

        #print(predictions_reshape.shape)
      #  print(controlpoints_fit.shape)
       # print(predictions.shape)
        # current_param_loss = loss_weights[0]*params_loss(predictions_reshape,controlpoints_fit)
       # print(predictions.shape)
       # print(fitpoints.shape)
        loss = kinematic_loss(predictions,fitpoints)
        # current_velocity_loss = loss_weights[2]*kinematic_loss(pred_vels/dt[:,None,None],fitvels)
        #loss = current_param_loss + current_position_loss + current_velocity_loss
        
        # Backward pass:
        optimizer.zero_grad()
        loss.backward() 
        # Weight and bias updates.
        optimizer.step()
        if use_tqdm:
            # logging information
            cum_loss += float(loss.item())
            num_samples += 1.0
            t.set_postfix({"cum_loss" : cum_loss/num_samples})
def go():
    parser = argparse.ArgumentParser(description="Train AdmiralNet Pose Predictor")
    parser.add_argument("dataset_config_file", type=str,  help="Dataset Configuration file to load")
    parser.add_argument("model_config_file", type=str,  help="Model Configuration file to load")
    parser.add_argument("output_directory", type=str,  help="Where to put the resulting model files")

    parser.add_argument("--context_length",  type=int, default=None,  help="Override the context length specified in the config file")
    parser.add_argument("--sequence_length",  type=int, default=None,  help="Override the sequence length specified in the config file")
    parser.add_argument("--epochstart", type=int, default=1,  help="Restart training from the given epoch number")
    parser.add_argument("--debug", action="store_true",  help="Display images upon each iteration of the training loop")
    parser.add_argument("--override", action="store_true",  help="Delete output directory and replace with new data")
    parser.add_argument("--tqdm", action="store_true",  help="Display tqdm progress bar on each epoch")
    parser.add_argument("--batch_size", type=int, default=None,  help="Override the order of the batch size specified in the config file")
    parser.add_argument("--gpu", type=int, default=None,  help="Override the GPU index specified in the config file")
    parser.add_argument("--learning_rate", type=float, default=None,  help="Override the learning rate specified in the config file")

    args = parser.parse_args()
    dataset_config_file = args.dataset_config_file
    config_file = args.model_config_file
    output_directory = os.path.join(args.output_directory,os.path.splitext(os.path.basename(config_file))[0])
    debug = args.debug
    epochstart = args.epochstart
    with open(config_file) as f:
        config = yaml.load(f, Loader = yaml.SafeLoader)

    with open(dataset_config_file) as f:
        dataset_config = yaml.load(f, Loader = yaml.SafeLoader)
    image_size = dataset_config["image_size"]
    input_channels = config["input_channels"]
    
    if args.context_length is not None:
        context_length = args.context_length
        config["context_length"]  = context_length
        output_directory+="_context%d"%(context_length)
    else:
        context_length = config["context_length"]

    if args.sequence_length is not None:
        sequence_length = args.sequence_length
        config["sequence_length"]  = sequence_length
        output_directory+="_sequence%d"%(sequence_length)
    else:
        sequence_length = config["sequence_length"]
    hidden_dimension = config["hidden_dimension"]
        
        
    #num_recurrent_layers = config["num_recurrent_layers"]
    if args.gpu is not None:
        gpu = args.gpu
        config["gpu"]  = gpu
    else:
        gpu = config["gpu"] 
    if args.batch_size is not None:
        batch_size = args.batch_size
        config["batch_size"]  = batch_size
    else:
        batch_size = config["batch_size"]
    if args.learning_rate is not None:
        learning_rate = args.learning_rate
        config["learning_rate"] = learning_rate
    else:
        learning_rate = config["learning_rate"]
    momentum = config["momentum"]
    num_epochs = config["num_epochs"]
    num_workers = config["num_workers"]
    use_float = config["use_float"]
    loss_weights = config["loss_weights"]
    print("Using config:\n%s" % (str(config)))
    net = nn_models.Models.AdmiralNetKinematicPredictor(context_length= context_length, sequence_length=sequence_length, input_channels=input_channels,\
        hidden_dim = hidden_dimension, output_dimension=2) 
    print("net:\n%s" % (str(net)))

    kinematic_loss = nn_models.LossFunctions.SquaredLpNormLoss()
    if use_float:
        print("casting stuff to float")
        net = net.float()
        kinematic_loss = kinematic_loss.float()
    else:
        print("casting stuff to double")
        net = net.double()
        kinematic_loss = kinematic_loss.float()
    if gpu>=0:
        print("moving stuff to GPU")
        net = net.cuda(gpu)
        kinematic_loss = kinematic_loss.cuda(gpu)
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
    
    dsets=[]
    use_optflow = net.input_channels==5
    for dataset in dataset_config["datasets"]:
        print("Parsing database config: %s" %(str(dataset)))
        label_folder = dataset["label_folder"]
        key_file = dataset["key_file"]
        image_folder = dataset["image_folder"]
        apply_color_jitter = dataset.get("apply_color_jitter",False)
        erasing_probability = dataset.get("erasing_probability",0.0)
        label_wrapper = deepracing.backend.PoseSequenceLabelLMDBWrapper()
        label_wrapper.readDatabase(os.path.join(label_folder,"lmdb"), max_spare_txns=max_spare_txns )
        image_mapsize = float(np.prod(image_size)*3+12)*float(len(label_wrapper.getKeys()))*1.1

        image_wrapper = deepracing.backend.ImageLMDBWrapper(direct_caching=False)
        image_wrapper.readDatabase(os.path.join(image_folder,"image_lmdb"), max_spare_txns=max_spare_txns, mapsize=image_mapsize )


        curent_dset = data_loading.proto_datasets.PoseSequenceDataset(image_wrapper, label_wrapper, key_file, context_length,\
                     image_size = image_size, return_optflow=use_optflow, apply_color_jitter=apply_color_jitter, erasing_probability=erasing_probability)
        dsets.append(curent_dset)
        print("\n")
    if len(dsets)==1:
        dset = dsets[0]
    else:
        dset = torch.utils.data.ConcatDataset(dsets)
    
    dataloader = data_utils.DataLoader(dset, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)
    print("Dataloader of of length %d" %(len(dataloader)))
    yaml.dump(dataset_config, stream=open(os.path.join(output_directory,"dataset_config.yaml"), "w"), Dumper = yaml.SafeDumper)
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
            run_epoch(net, optimizer, dataloader, gpu, kinematic_loss, loss_weights, debug=debug, use_tqdm=args.tqdm, use_float = use_float)
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
    