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
import torchvision.transforms as transforms
def run_epoch(network, criterion, optimizer, trainLoader, gpu = -1):
    network.train()  # This is important to call before training!
    cum_loss = 0.0
    batch_size = trainLoader.batch_size
    num_samples=0
    t = tqdm(enumerate(trainLoader))
    for (i, (inputs, labels)) in t:
        if gpu>=0:
            inputs = inputs.cuda(gpu)
            labels = labels.cuda(gpu)
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
 

def train_model(network, criterion, optimizer, trainLoader, file_prefix, directory, n_epochs = 10, gpu = -1):
    if gpu>=0:
        criterion = criterion.cuda(gpu)
    # Training loop.
    if(not os.path.isdir(directory)):
        os.makedirs(directory)
    for epoch in range(0, n_epochs):
        print("Epoch %d of %d" %((epoch+1),n_epochs))
        run_epoch(network, criterion, optimizer, trainLoader, gpu)
        log_path = os.path.join(directory,""+file_prefix+"_epoch"+str((epoch+1))+ ".model")
        torch.save(network.state_dict(), log_path)
def load_config(filepath):
    config_file = open(filepath)
    lines = config_file.readlines()
    if len(lines)==0:
        return dict()
    vals = []
    for line in lines:
        key, value = line.split(",")
        key = key.replace("\n","")
        value = value.replace("\n","")
        vals.append((key,value))
    return dict(vals)
def main():
    parser = argparse.ArgumentParser(description="Steering prediction with PilotNet")
    parser.add_argument("--config_file", type=str, required=True, help="Config file to use")
    args = parser.parse_args()
    config = load_config(args.config_file)
    print("Overwriting these config parameters.", config)

    
    learning_rate = float(config['learning_rate'])
    root_dir, annotation_file = os.path.split(config['annotation_file'])
    prefix, _ = annotation_file.split(".")
    output_dir = config['output_dir']

    batch_size = int(config.get('batch_size','1'))
    gpu = int(config.get('gpu','-1'))
    epochs = int(config.get('epochs','10'))
    momentum = float(config.get('momentum','0.0'))
    file_prefix = config.get('file_prefix','')
    load_files = bool(config.get('load_files',''))
    use_float32 = bool(config.get('use_float32',''))
    context_length = int(config.get('context_length','10'))
    sequence_length = int(config.get('sequence_length','5'))
    hidden_dim = int(config.get('hidden_dim','100'))
    label_scale = float(config.get('label_scale','100.0'))
    workers = int(config.get('workers','0'))

    
    prefix = prefix + file_prefix
    network = models.AdmiralNet(context_length = context_length, sequence_length=sequence_length, hidden_dim = hidden_dim, use_float32 = use_float32, gpu = gpu)
    img_transformation = transforms.Compose([transforms.Lambda(lambda inputs: inputs.div(255.0)), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    if(label_scale == 1.0):
        label_transformation = None
    else:
        label_transformation = transforms.Compose([transforms.Lambda(lambda inputs: inputs.mul(label_scale))])
    if(use_float32):
        network.float()
        trainset = loaders.F1SequenceDataset(root_dir,annotation_file,(66,200),\
        context_length=context_length, sequence_length=sequence_length, use_float32=True, img_transformation = img_transformation, label_transformation = label_transformation)
    else:
        network.double()
        trainset = loaders.F1SequenceDataset(root_dir, annotation_file,(66,200),\
        context_length=context_length, sequence_length=sequence_length, img_transformation = img_transformation, label_transformation = label_transformation)
    if(gpu>=0):
        network = network.cuda(gpu)
    
    
   # trainset.read_files()
    
    if(load_files or (not os.path.isfile("./" + prefix+"_images.pkl")) or (not os.path.isfile("./" + prefix+"_annotations.pkl"))):
        trainset.read_files()
        trainset.write_pickles(prefix+"_images.pkl",prefix+"_annotations.pkl")
    else:  
        trainset.read_pickles(prefix+"_images.pkl",prefix+"_annotations.pkl")
    ''' '''
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = workers)
    print(trainLoader)
    #Definition of our loss.
    criterion = nn.MSELoss()

    # Definition of optimization strategy.
    optimizer = optim.SGD(network.parameters(), lr = learning_rate, momentum=momentum)
    train_model(network, criterion, optimizer, trainLoader, prefix, output_dir, n_epochs = epochs, gpu = gpu)

if __name__ == '__main__':
    main()