import cv2
import numpy as np
import nn_models
import data_loading.image_loading as il
import nn_models.Models as models
import data_loading.data_loaders as loaders
import numpy.random
import torch, random
import torch.nn as nn 
import torch.optim as optim
from tqdm import tqdm as tqdm
import pickle
from datetime import datetime
import os
import string
import argparse
def run_epoch(network, criterion, optimizer, trainLoader, use_gpu):
    network.train()  # This is important to call before training!
    cum_loss = 0.0
    batch_size = trainLoader.batch_size
    num_samples=0
    t = tqdm(enumerate(trainLoader))
    for (i, (inputs, labels)) in t:
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        # Forward pass:
        outputs = network(inputs)
        loss = criterion(outputs, labels)

        # Backward pass:
        optimizer.zero_grad()
        loss.backward() 

        # Weight and bias updates.
        optimizer.step()

        # logging information.
        cum_loss += loss.item()
        num_samples += batch_size
        t.set_postfix(cum_loss = cum_loss/num_samples)
 

def train_model(network, criterion, optimizer, trainLoader, file_prefix, directory, n_epochs = 10, use_gpu = False):
    if use_gpu:
        network = network.cuda()
        criterion = criterion.cuda()
    # Training loop.
    if(not os.path.isdir(directory)):
        os.makedirs(directory)
    for epoch in range(0, n_epochs):
        print("Epoch %d of %d" %((epoch+1),n_epochs))
        run_epoch(network, criterion, optimizer, trainLoader, use_gpu)
        log_path = os.path.join(directory,""+file_prefix+"_epoch"+str((epoch+1))+ ".model")
        torch.save(network.state_dict(), log_path)
def main():
    parser = argparse.ArgumentParser(description="Steering prediction with PilotNet")
    parser.add_argument("--gpu", action="store_true", help="Accelerate with GPU")
    parser.add_argument("--batch_size", type=int, default = 8, help="Batch Size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs to run")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Number of training epochs to run")
    parser.add_argument("--momentum", type=float, default=0.0, help="Momentum value to use on the SGD optimizer")
    parser.add_argument("--root_dir", type=str, required=True, help="Root dir of the F1 dataset to use")
    parser.add_argument("--annotation_file", type=str, required=True, help="Annotation file to use")
    parser.add_argument("--output_dir", type=str, default="log", help="Directory to place the model files")
    parser.add_argument("--file_prefix", type=str, default="", help="Additional prefix to add to the filename for the saved weight files")
    args = parser.parse_args()
    batch_size = args.batch_size
    prefix, ext = args.annotation_file.split(".")
    prefix ="captain_net_" + prefix + args.file_prefix
    network = models.CaptainNet()
    network.float()

    trainset = loaders.F1AllControlDataset(args.root_dir,args.annotation_file,(3,66,200))
   # trainset.read_files()
    
    if(os.path.isfile("./" + prefix+"_images.pkl") and os.path.isfile("./" + prefix+"_annotations.pkl")):
        trainset.read_pickles(prefix+"_images.pkl",prefix+"_annotations.pkl")
    else:  
        trainset.read_files(use_float32 = True)
        trainset.write_pickles(prefix+"_images.pkl",prefix+"_annotations.pkl")
    ''' '''
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 0)
    print(trainLoader)
    #Definition of our loss.
    criterion = nn.L1Loss()

    # Definition of optimization strategy.
    optimizer = optim.SGD(network.parameters(), lr = args.learning_rate, momentum=args.momentum)
    train_model(network, criterion, optimizer, trainLoader, prefix, args.output_dir, n_epochs = args.epochs, use_gpu = args.gpu)

if __name__ == '__main__':
    main()