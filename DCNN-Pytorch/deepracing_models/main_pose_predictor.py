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
def run_epoch(network, optimizer, trainLoader, gpu, position_loss, rotation_loss, loss_weights=[1.0, 1.0], imsize=(66,200), debug=False, use_tqdm=True, use_float=True):
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
    network.train()  # This is important to call before training!
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
        if(opt_flow_torch is not None):
            image_torch =torch.cat((image_torch,opt_flow_torch),axis=1)
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
    parser.add_argument("output_directory", type=str,  help="Where to put the resulting model files")

    parser.add_argument("--debug", action="store_true",  help="Display images upon each iteration of the training loop")
    parser.add_argument("--override", action="store_true",  help="Delete output directory and replace with new data")
    parser.add_argument("--tqdm", action="store_true",  help="Display tqdm progress bar on each epoch")
    args = parser.parse_args()
    config_file = args.config_file
    output_directory = args.output_directory
    debug = args.debug
    with open(config_file) as f:
        config = yaml.load(f, Loader = yaml.SafeLoader)
    image_db = config["image_db"]
    opt_flow_db = config["opt_flow_db"]
    label_db = config["label_db"]
    key_file = config["key_file"]
    image_size = config["image_size"]
    hidden_dimension = config["hidden_dimension"]
    input_channels = config["input_channels"]
    sequence_length = config["sequence_length"]
    context_length = config["context_length"]
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
    if (not args.override) and os.path.isdir(output_directory):
        s = ""
        while(not (s=="y" or s=="n")):
             s = input("Directory " + output_directory + " already exists. Overwrite it with new data? [y\\n]\n")
        if s=="n":
            print("Thanks for playing!")
            exit(0)
        shutil.rmtree(output_directory)
    elif os.path.isdir(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)
    net = models.AdmiralNetPosePredictor(gpu=gpu,context_length = context_length, sequence_length = sequence_length,\
        hidden_dim=hidden_dimension, input_channels=input_channels, temporal_conv_feature_factor = temporal_conv_feature_factor)
    
    position_loss = torch.nn.MSELoss(reduction=position_loss_reduction)
    rotation_loss = loss_functions.QuaternionDistance()
    optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum=momentum)
    if use_float:
        net = net.float()
        position_loss = position_loss.float()
        rotation_loss = rotation_loss.float()
    else:
        net = net.double()
        position_loss = position_loss.double()
        rotation_loss = rotation_loss.double()
    if gpu>=0:
        rotation_loss = rotation_loss.cuda(gpu)
        position_loss = position_loss.cuda(gpu)
        net = net.cuda(gpu)
    if num_workers == 0:
        max_spare_txns = 1
    else:
        max_spare_txns = num_workers

    #image_wrapper = deepracing.backend.ImageFolderWrapper(os.path.dirname(image_db))

    label_wrapper = deepracing.backend.PoseSequenceLabelLMDBWrapper()
    label_wrapper.readDatabase(label_db, max_spare_txns=max_spare_txns )

    image_size = np.array(image_size)
    image_mapsize = float(np.prod(image_size)*3+12)*float(len(label_wrapper.getKeys()))*1.1
    image_wrapper = deepracing.backend.ImageLMDBWrapper()
    image_wrapper.readDatabase(image_db, max_spare_txns=max_spare_txns, mapsize=image_mapsize )
    optical_flow_db_wrapper = deepracing.backend.OpticalFlowLMDBWrapper()
    optical_flow_db_wrapper.readDatabase(opt_flow_db, max_spare_txns=max_spare_txns, mapsize=int(round( float(image_mapsize)*8/3) ) )

    dset = data_loading.proto_datasets.PoseSequenceDataset(image_wrapper, label_wrapper, key_file, context_length, sequence_length, image_size = image_size, optical_flow_db_wrapper=optical_flow_db_wrapper)
    dataloader = data_utils.DataLoader(dset, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)
    print("Dataloader of of length %d" %(len(dataloader)))
    yaml.dump(config, stream=open(os.path.join(output_directory,"config.yaml"), "w"), Dumper = yaml.SafeDumper)
    i = 0
    while i < num_epochs:
        time.sleep(2.0)
        postfix = i + 1
        print("Running Epoch Number %d" %(postfix))
        #dset.clearReaders()
        try:
            tick = time.time()
            run_epoch(net, optimizer, dataloader, gpu, position_loss, rotation_loss, loss_weights=loss_weights, debug=debug, use_tqdm=args.tqdm, use_float = use_float)
            tock = time.time()
            print("Finished epoch %d in %f seconds." % ( postfix , tock-tick ) )
        except Exception as e:
            print("Restarting epoch %d because %s"%(postfix, str(e)))
            modelin = os.path.join(output_directory,"epoch_%d_params.pt" %(postfix-1))
            optimizerin = os.path.join(output_directory,"epoch_%d_optimizer.pt" %(postfix-1))
            net.load_state_dict(torch.load(modelin))
            optimizer.load_state_dict(torch.load(optimizerin))
            continue
        modelout = os.path.join(output_directory,"epoch_%d_params.pt" %(postfix))
        torch.save(net.state_dict(), modelout)
        optimizerout = os.path.join(output_directory,"epoch_%d_optimizer.pt" %(postfix))
        torch.save(optimizer.state_dict(), optimizerout)
        irand = np.random.randint(0,high=len(dset))
        imtest = torch.rand( 1, context_length, input_channels, image_size[0], image_size[1], dtype=torch.float32 )
        imtest[0], positions_torch, quats_torch, _, _, _ = dset[irand]
        if use_float:
            imtest = imtest.float()
        else:
            imtest = imtest.double()
        if(gpu>=0):
            imtest = imtest.cuda(gpu)
        pos_pred, rot_pred = net(imtest)
        print(positions_torch)
        print(pos_pred)
        print(quats_torch)
        print(rot_pred)
        i = i + 1
import logging
if __name__ == '__main__':
    logging.basicConfig()
    go()    
    