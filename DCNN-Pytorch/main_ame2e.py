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


loss = torch.zeros(1)
def run_epoch(network, optimizer, trainLoader, gpu, loss_function, imsize=(66,200), debug=False, use_tqdm=True):
    global loss
    cum_loss = 0.0
    batch_size = trainLoader.batch_size
    num_samples=0.0
    if use_tqdm:
        t = tqdm(enumerate(trainLoader), total=len(trainLoader))
    else:
        t = enumerate(trainLoader)
    network.train()  # This is important to call before training!
    for (i, (image_torch, optflow_torch, control_output) ) in t:
        if debug:
            image_np = image_torch[0].numpy().copy().transpose(1,2,0)
            image_ubyte = skimage.util.img_as_ubyte(image_np)
            print(control_output[0])
            cv2.namedWindow("Image",cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Image", cv2.cvtColor(image_ubyte,cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
        image_torch = image_torch.double()
        optflow_torch = optflow_torch.double()
        control_output = control_output.double()

        if gpu>=0:
            image_torch = image_torch.cuda(gpu)
            optflow_torch = optflow_torch.cuda(gpu)
            control_output = control_output.cuda(gpu)
      #  print(image_torch.dtype)
        # Forward pass:
        if network.input_channels==5:
            input_torch = torch.cat((image_torch, optflow_torch), dim=2)
        elif network.input_channels==3:
            input_torch = image_torch
        elif network.input_channels==2:
            input_torch = optflow_torch
        else:
            raise ValueError("Invalid number of input channels: %d" %(net.input_channels))
        predictions = network(input_torch)
        loss = loss_function(predictions, control_output)
        
        # Backward pass:
        optimizer.zero_grad()
        loss.backward() 

        # Weight and bias updates.
        optimizer.step()

        if use_tqdm:
            # logging information
            cum_loss += float(loss.item())
            num_samples += float(batch_size)
            t.set_postfix({"cum_loss" : cum_loss/num_samples})
def go():
    parser = argparse.ArgumentParser(description="Train AdmiralNet Control Predictor")
    parser.add_argument("training_config", type=str,  help="Training Parameters Configuration file to load")
    parser.add_argument("dataset_config", type=str,  help="Dataset Configuration file to load")
    parser.add_argument("output_directory", type=str,  help="Where to put the resulting model files")

    parser.add_argument("--debug", action="store_true",  help="Display images upon each iteration of the training loop")
    parser.add_argument("--tqdm", action="store_true",  help="Display tqdm progress bar on each epoch")
    parser.add_argument("--gpu", type=int, default=None,  help="Override GPU number in config file")
    parser.add_argument("--batch_size", type=int, default=None,  help="Override the order of the batch size specified in the config file")
    parser.add_argument("--learning_rate", type=float, default=None,  help="Override the learning rate specified in the config file")
    parser.add_argument("--momentum", type=float, default=None,  help="Override the momentum specified in the config file")
    parser.add_argument("--loss_function", type=str, default=None,  help="Override the loss function specified in the config file")

    
    args : argparse.Namespace = parser.parse_args()
    
    argsdict = dict(args.__dict__)
    print(argsdict)
    training_config_file = args.training_config
    dataset_config_file = args.dataset_config
    debug = args.debug
    with open(training_config_file) as f:
        config = yaml.load(f, Loader = yaml.SafeLoader)
    with open(dataset_config_file) as f:
        dataset_config = yaml.load(f, Loader = yaml.SafeLoader)

    image_size = dataset_config["image_size"]
    output_dimension = config["output_dimension"]
    sequence_length = config["sequence_length"]
    context_length = config["context_length"]
    hidden_dimension = config["hidden_dimension"]
    input_channels  = config["input_channels"] 
    if args.gpu is not None:
        gpu = args.gpu
        config["gpu"]=gpu
    else:
        gpu = config["gpu"]

    if args.momentum is not None:
        momentum = args.momentum
        config["momentum"]=momentum
    else:
        momentum = config["momentum"]

    if args.batch_size is not None:
        batch_size = args.batch_size
        config["batch_size"]=batch_size
    else:
        batch_size = config["batch_size"]

    if args.learning_rate is not None:
        learning_rate = args.learning_rate
        config["learning_rate"]=learning_rate
    else:
        learning_rate = config["learning_rate"]

    if args.loss_function is not None:
        loss_function = args.loss_function
        config["loss_function"]=loss_function
    else:
        loss_function = config["loss_function"]

    num_epochs = config["num_epochs"]
    num_workers = config["num_workers"]
    loss_reduction = config["loss_reduction"]
    nesterov = config["nesterov"]

    
    net = deepracing_models.nn_models.Models.AdmiralNetKinematicPredictor(input_channels=input_channels, output_dimension=2, \
                                                    context_length=context_length, sequence_length=sequence_length, hidden_dim = hidden_dimension)
    if loss_function=="L1":
        loss_func = torch.nn.L1Loss(reduction=loss_reduction)
    elif loss_function=="MSE":
        loss_func = torch.nn.MSELoss(reduction=loss_reduction)
    else:
        raise ValueError("Unknown loss function: " + loss_function)

    net = net.double()
    loss_func = loss_func.double()
    if gpu>=0:
        loss_func = loss_func.cuda(gpu)
        net = net.cuda(gpu)
    if num_workers == 0:
        max_spare_txns = 16
    else:
        max_spare_txns = num_workers
    optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum=momentum, dampening=0.0, nesterov=(nesterov and momentum>0) )
    main_dir = args.output_directory
    experiment = comet_ml.Experiment(project_name="deepracingadmiralnet-e2e", workspace="electric-turtle")
    experiment.log_parameters(config)
    experiment.log_parameters(dataset_config)
    experiment.add_tag("bezierpredictor")
    experiment_config = {"experiment_key": experiment.get_key()}
    output_directory = os.path.join(main_dir, experiment.get_key())
    if os.path.isdir(output_directory) :
        raise FileExistsError( "%s already exists, this should not happen." % (output_directory) )
    os.makedirs(output_directory)
    yaml.dump(experiment_config, stream=open(os.path.join(output_directory,"experiment_config.yaml"),"w"), Dumper=yaml.SafeDumper)
    yaml.dump(dataset_config, stream=open(os.path.join(output_directory,"dataset_config.yaml"), "w"), Dumper = yaml.SafeDumper)
    yaml.dump(config, stream=open(os.path.join(output_directory,"model_config.yaml"), "w"), Dumper = yaml.SafeDumper)
    datasets = dataset_config["datasets"]
    dsets=[]
    use_optflow=True
    for dataset in datasets:
        print("Parsing database config: %s" %(str(dataset)))
        root_folder = dataset["root_folder"]
        image_folder = os.path.join(root_folder,"images")
        image_lmdb = os.path.join(image_folder,"image_lmdb")
        optflow_lmdb = os.path.join(image_folder,"optical_flow_lmdb")
        label_folder = os.path.join(root_folder,"steering_labels")
        label_lmdb = os.path.join(label_folder,"lmdb")
        key_file = os.path.join(root_folder,dataset["key_file"])

        label_wrapper = deepracing.backend.ControlLabelLMDBWrapper()
        label_wrapper.readDatabase(label_lmdb, mapsize=1e9, max_spare_txns=max_spare_txns )

        image_size = np.array(image_size)
        image_mapsize = float(np.prod(image_size)*3+12)*float(len(label_wrapper.getKeys()))*1.1
        image_wrapper = deepracing.backend.ImageLMDBWrapper(direct_caching=False)
        image_wrapper.readDatabase(image_lmdb, max_spare_txns=max_spare_txns, mapsize=image_mapsize )

        optflow_wrapper = deepracing.backend.OpticalFlowLMDBWrapper()
        optflow_wrapper.readDatabase(optflow_lmdb, mapsize=8*image_mapsize)
        
        curent_dset = PD.ControlOutputSequenceDataset(image_wrapper, label_wrapper, key_file,\
             context_length=context_length, sequence_length=sequence_length, image_size = image_size, optflow_db_wrapper=optflow_wrapper)
        dsets.append(curent_dset)
    if len(dsets)==1:
        dset = dsets[0]
    else:
        dset = torch.utils.data.ConcatDataset(dsets)
    
    dataloader = data_utils.DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=gpu>=0)
    print("Dataloader of of length %d" %(len(dataloader)))
    i = 0
    netpostfix="admiralnet_epoch_%d_params.pt" 
    optimizerpostfix = "admiralnet_epoch_%d_optimizer.pt"
    with experiment.train():
        while i < num_epochs:
            time.sleep(2.0)
            postfix = i + 1
            print("Running Epoch Number %d" %(postfix))
            #dset.clearReaders()
            try:
                tick = time.time()
                run_epoch(net, optimizer, dataloader, gpu, loss_func, debug=debug, use_tqdm=args.tqdm)
                tock = time.time()
                print("Finished epoch %d in %f seconds." % ( postfix , tock-tick ) )
            except Exception as e:
                print("Restarting epoch %d because %s"%(postfix, str(e)))
                modelin = os.path.join(output_directory,netpostfix %(postfix-1))
                optimizerin = os.path.join(output_directory,optimizerpostfix %(postfix-1))
                net.load_state_dict(torch.load(modelin))
                optimizer.load_state_dict(torch.load(optimizerin))
                continue

            netfile = netpostfix % ( postfix )
            modelout = os.path.join( output_directory, netfile )
            torch.save( net.state_dict(), modelout )
            experiment.log_asset(modelout, file_name=netfile)
            
            optimizerfile = optimizerpostfix % ( postfix )
            optimizerout = os.path.join( output_directory, optimizerfile )
            torch.save( optimizer.state_dict(), optimizerout )
            experiment.log_asset(optimizerout, file_name=optimizerfile )
            i = i + 1
import logging
if __name__ == '__main__':
    logging.basicConfig()
    go()    
    