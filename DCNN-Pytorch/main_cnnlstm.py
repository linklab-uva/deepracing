import comet_ml
import torch
import torch.utils.data as data_utils
import deepracing_models.data_loading.proto_datasets
from tqdm import tqdm as tqdm
import deepracing_models.nn_models.LossFunctions as loss_functions
import deepracing_models.nn_models.Models as models
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
import cv2
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
    for (i, (image_torch, control_output) ) in t:
        if debug:
            image_np = image_torch[0].numpy().copy().transpose(1,2,0)
            image_ubyte = skimage.util.img_as_ubyte(image_np)
            print(control_output[0])
            cv2.namedWindow("Image",cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Image", cv2.cvtColor(image_ubyte,cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
        image_torch = image_torch.double()
        control_output = control_output.double()
        if gpu>=0:
            image_torch = image_torch.cuda(gpu)
            control_output = control_output.cuda(gpu)
        # Forward pass:
        predictions = network(image_torch)
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
    parser = argparse.ArgumentParser(description="Train CNNLSTM Control Predictor")
    parser.add_argument("training_config", type=str,  help="Training Parameters Configuration file to load")
    parser.add_argument("dataset_config", type=str,  help="Dataset Configuration file to load")
    parser.add_argument("output_directory", type=str,  help="Where to put the resulting model files")

    parser.add_argument("--debug", action="store_true",  help="Display images upon each iteration of the training loop")
    parser.add_argument("--override", action="store_true",  help="Delete output directory and replace with new data")
    parser.add_argument("--tqdm", action="store_true",  help="Display tqdm progress bar on each epoch")
    parser.add_argument("--epochstart", type=int, default=1,  help="Restart training from the given epoch number")
    parser.add_argument("--gpu", type=int, default=-1,  help="Override GPU number in config file")
    args = parser.parse_args()
    epochstart = args.epochstart
    training_config_file = args.training_config
    dataset_config_file = args.dataset_config
    output_directory = os.path.join(args.output_directory,os.path.splitext(os.path.basename(training_config_file))[0])
    debug = args.debug
    with open(training_config_file) as f:
        training_config = yaml.load(f, Loader = yaml.SafeLoader)
    with open(dataset_config_file) as f:
        dataset_config = yaml.load(f, Loader = yaml.SafeLoader)

    image_size = dataset_config["image_size"]
    input_channels = training_config["input_channels"]
    output_dimension = training_config["output_dimension"]
    if args.gpu>=0:
        gpu = args.gpu
    else:
        gpu = training_config["gpu"]
    batch_size = training_config["batch_size"]
    learning_rate = training_config["learning_rate"]
    momentum = training_config["momentum"]
    num_epochs = training_config["num_epochs"]
    num_workers = training_config["num_workers"]
    control_loss_reduction = training_config["control_loss_reduction"]

    
    context_length = training_config["context_length"]
    sequence_length = training_config["sequence_length"]
    if (not args.override) and os.path.isdir(output_directory):
        s = ""
        while(not (s=="y" or s=="n")):
             s = input( "Directory %s already exists. Overwrite it with new data? [y\\n]\n" % (output_directory) )
        if s=="n":
            print("Thanks for playing!")
            exit(0)
        shutil.rmtree(output_directory)
    elif os.path.isdir(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)
    net = models.CNNLSTM(input_channels=input_channels, output_dimension = output_dimension, context_length = context_length, sequence_length=sequence_length)
    
    mse_loss = torch.nn.MSELoss(reduction=control_loss_reduction)
    optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum=momentum)
    net = net.double()
    mse_loss = mse_loss.double()
    if gpu>=0:
        mse_loss = mse_loss.cuda(gpu)
        net = net.cuda(gpu)
    if num_workers == 0:
        max_spare_txns = 16
    else:
        max_spare_txns = num_workers
    datasets = dataset_config["datasets"]
    dsets=[]
    use_optflow=True
    for dataset in datasets:
        print("Parsing database config: %s" %(str(dataset)))
        image_folder = dataset["image_folder"]
        image_lmdb = os.path.join(image_folder,"image_lmdb")
        label_folder = dataset["label_folder"]
        label_lmdb = os.path.join(label_folder,"lmdb")
        key_file = dataset["key_file"]

        label_wrapper = deepracing.backend.ControlLabelLMDBWrapper()
        label_wrapper.readDatabase(label_lmdb, mapsize=3e9, max_spare_txns=max_spare_txns )

        image_size = np.array(image_size)
        image_mapsize = float(np.prod(image_size)*3+12)*float(len(label_wrapper.getKeys()))*1.1
        image_wrapper = deepracing.backend.ImageLMDBWrapper(direct_caching=False)
        image_wrapper.readDatabase(image_lmdb, max_spare_txns=max_spare_txns, mapsize=image_mapsize )
        
        curent_dset = deepracing_models.data_loading.proto_datasets.ControlOutputSequenceDataset(image_wrapper, label_wrapper, key_file, image_size = image_size, context_length = context_length, sequence_length=sequence_length)
        dsets.append(curent_dset)
    if len(dsets)==1:
        dset = dsets[0]
    else:
        dset = torch.utils.data.ConcatDataset(dsets)
    
    dataloader = data_utils.DataLoader(dset, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)
    print("Dataloader of of length %d" %(len(dataloader)))
    yaml.dump(training_config, stream=open(os.path.join(output_directory,"training_config.yaml"), "w"), Dumper = yaml.SafeDumper)
    yaml.dump(dataset_config, stream=open(os.path.join(output_directory,"dataset_config.yaml"), "w"), Dumper = yaml.SafeDumper)
    experiment = comet_ml.Experiment(workspace="electric-turtle", project_name="deepracingcnnlstm")
    experiment.log_parameters(training_config)
    experiment.log_parameters(dataset_config)
    experiment_config = {"experiment_key": experiment.get_key()}
    yaml.dump(experiment_config, stream=open(os.path.join(output_directory,"experiment_config.yaml"),"w"), Dumper=yaml.SafeDumper)
    i = 0
    netpostfix="CNNLSTM_epoch_%d_params.pt" 
    optimizerpostfix = "CNNLSTM_epoch_%d_optimizer.pt"
    with experiment.train():
        while i < num_epochs:
            time.sleep(2.0)
            postfix = i + 1
            print("Running Epoch Number %d" %(postfix))
            #dset.clearReaders()
            try:
                tick = time.time()
                run_epoch(net, optimizer, dataloader, gpu, mse_loss, debug=debug, use_tqdm=args.tqdm)
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

            irand = np.random.randint(0,high=len(dset))
            input_test = torch.rand( 1, context_length, input_channels, image_size[0], image_size[1], dtype=torch.float32 )
            input_test[0], control_label = dset[irand]
            input_test = input_test.double()
            if gpu>=0:
                input_test = input_test.cuda(gpu)
            control_pred = net(input_test)
            print(control_pred)
            print(control_label)

            i = i + 1
import logging
if __name__ == '__main__':
    logging.basicConfig()
    go()    
    