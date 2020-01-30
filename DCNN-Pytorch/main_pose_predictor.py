import comet_ml
import torch
import torch.nn as NN
import torch.utils.data as data_utils
import deepracing_models.data_loading.proto_datasets as PD
from tqdm import tqdm as tqdm
import deepracing_models.nn_models
import deepracing_models.nn_models.Models
import deepracing_models.nn_models.LossFunctions as loss_functions
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
import deepracing_models.math_utils.bezier
#torch.backends.cudnn.enabled = False
def run_epoch(experiment, network, optimizer, trainLoader, gpu, kinematic_loss, loss_weights, imsize=(66,200), timewise_weights=None, debug=False, use_tqdm=True, use_float=True):
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
        numpoints = positions_torch.shape[1]
        predictions = network(image_torch)
        pointsx = positions_torch[:,:,0]
        pointsz = positions_torch[:,:,2]
        pointsx_interp = torch.nn.functional.interpolate(pointsx.view(-1,1,numpoints), size=network.sequence_length, scale_factor=None, mode='linear', align_corners=None).squeeze()
        pointsz_interp = torch.nn.functional.interpolate(pointsz.view(-1,1,numpoints), size=network.sequence_length, scale_factor=None, mode='linear', align_corners=None).squeeze()
        gtpoints = torch.stack([pointsx_interp,pointsz_interp],dim=1).transpose(1,2)
       # print(fitpoints.shape)
        if debug:
            fig = plt.figure()
            ax = fig.add_subplot()
            pointsxnp = pointsx[0].detach().cpu().numpy().copy()
            pointsznp = pointsz[0].detach().cpu().numpy().copy()
            gtpointsnp = gtpoints[0].detach().cpu().numpy().copy()
            ax.plot(pointsxnp,pointsznp,'r-')
            ax.plot(gtpointsnp[:,0],gtpointsnp[:,1],'b+')
            
            plt.show()

        #print(predictions_reshape.shape)
      #  print(controlpoints_fit.shape)
       # print(predictions.shape)
        # current_param_loss = loss_weights[0]*params_loss(predictions_reshape,controlpoints_fit)
       # print(predictions.shape)
       # print(fitpoints.shape)
        loss = kinematic_loss(predictions,gtpoints)
        # current_velocity_loss = loss_weights[2]*kinematic_loss(pred_vels/dt[:,None,None],fitvels)
        #loss = current_param_loss + current_position_loss + current_velocity_loss
        
        # Backward pass:
        optimizer.zero_grad()
        loss.backward() 
        # Weight and bias updates.
        optimizer.step()
        # logging information
        cum_loss += float(loss.item())
        num_samples += 1.0
        cumulative_average_loss = cum_loss/num_samples
        experiment.log_metric("cumulative_average_loss", cumulative_average_loss, step=i)
        if use_tqdm:
            t.set_postfix({"cumulative_average_loss" : cumulative_average_loss})
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
    net = deepracing_models.nn_models.Models.AdmiralNetKinematicPredictor(context_length= context_length, sequence_length=sequence_length, input_channels=input_channels, hidden_dim = hidden_dimension, output_dimension=2) 
    print("net:\n%s" % (str(net)))

    kinematic_loss = loss_functions.SquaredLpNormLoss()
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
    main_dir = args.output_directory
    experiment = comet_ml.Experiment(workspace="electric-turtle", project_name="deepracingposepredictor")
    experiment.log_parameters(config)
    experiment.log_parameters(dataset_config)
    experiment.add_tag("bezierpredictor")
    experiment_config = {"experiment_key": experiment.get_key()}
    output_directory = os.path.join(main_dir, experiment.get_key())
    if os.path.isdir(output_directory) :
        raise FileExistsError("%s already exists, this should not happen." %(output_directory) )
    os.makedirs(output_directory)
    
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


        curent_dset = PD.PoseSequenceDataset(image_wrapper, label_wrapper, key_file, context_length,\
                     image_size = image_size, return_optflow=use_optflow, apply_color_jitter=apply_color_jitter, erasing_probability=erasing_probability)
        dsets.append(curent_dset)
        print("\n")
    if len(dsets)==1:
        dset = dsets[0]
    else:
        dset = torch.utils.data.ConcatDataset(dsets)
    
    dataloader = data_utils.DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=gpu>=0)
    print("Dataloader of of length %d" %(len(dataloader)))
    yaml.dump(experiment_config, stream=open(os.path.join(output_directory,"experiment_config.yaml"),"w"), Dumper=yaml.SafeDumper)
    yaml.dump(dataset_config, stream=open(os.path.join(output_directory,"dataset_config.yaml"), "w"), Dumper = yaml.SafeDumper)
    yaml.dump(config, stream=open(os.path.join(output_directory,"model_config.yaml"), "w"), Dumper = yaml.SafeDumper)
    i = 0
    netpostfix = "epoch_%d_params.pt"
    optimizerpostfix = "epoch_%d_optimizer.pt"
    with experiment.train():
        while i < num_epochs:
            time.sleep(2.0)
            postfix = i + 1
            print("Running Epoch Number %d" %(postfix))
            #dset.clearReaders()
            tick = time.time()
            run_epoch(experiment, net, optimizer, dataloader, gpu, kinematic_loss, loss_weights, debug=debug, use_tqdm=args.tqdm, use_float = use_float)
            tock = time.time()
            print("Finished epoch %d in %f seconds." % ( postfix , tock-tick ) )
            modelout = os.path.join(output_directory,netpostfix %(postfix))
            torch.save(net.state_dict(), modelout)
            with open(modelout,'rb') as modelfile:
                experiment.log_asset(modelfile,file_name=netpostfix %(postfix))
            optimizerout = os.path.join(output_directory,optimizerpostfix %(postfix))
            torch.save(optimizer.state_dict(), optimizerout)
            with open(optimizerout,'rb') as optimizerfile:
                experiment.log_asset(optimizerfile,file_name=optimizerpostfix %(postfix))
            
            i = i + 1
import logging
if __name__ == '__main__':
    logging.basicConfig()
    go()    
    