import cv2
import numpy as np
import nn_models
import data_loading.image_loading as il
import nn_models.Models as models
import data_loading.data_loaders_old as loaders
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
def run_epoch(network, criterion, optimizer, trainLoader, use_gpu):
    network.train()  # This is important to call before training!
    cum_loss = 0.0
    batch_size = trainLoader.batch_size
    num_samples=0
    t = tqdm(enumerate(trainLoader))
    for (i, (inputs, throttle, brake, labels)) in t:
        optimizer.zero_grad()
        if use_gpu>=0:
            inputs = inputs.cuda(use_gpu)
            labels = labels.cuda(use_gpu)
        # Forward pass:
        #print(labels.size())
        outputs = network(inputs)
        loss = criterion(outputs, labels)

        # Backward pass:
        loss.backward() 

        # Weight and bias updates.
        optimizer.step()

        # logging information
        loss_ = loss.item()
        cum_loss += loss_
        num_samples += batch_size
        t.set_postfix(cum_loss = cum_loss/num_samples)
    return cum_loss/num_samples
 

def train_model(network, criterion, optimizer, trainLoader, file_prefix, directory, n_epochs = 10, use_gpu = False, starting_epoch = 0):
    if use_gpu>=0:
        criterion = criterion.cuda(use_gpu)
    # Training loop.
    if(not os.path.isdir(directory)):
        os.makedirs(directory)
    losses = []
    for epoch in range(starting_epoch, starting_epoch + n_epochs):
        print("Epoch %d of %d" %((starting_epoch + epoch+1),n_epochs))
        loss = run_epoch(network, criterion, optimizer, trainLoader, use_gpu)
        losses.append(loss)
        log_path = os.path.join(directory,""+file_prefix+"_epoch"+str((starting_epoch + epoch+1))+ ".model")
        torch.save(network.state_dict(), log_path)
    return losses
def load_config(filepath):
    rtn = dict()
    rtn['batch_size']='1'
    rtn['gpu']='-1'
    rtn['epochs']='10'
    rtn['momentum']='0.0'
    rtn['file_prefix']=''
    rtn['load_files']=''
    rtn['use_float32']=''
    rtn['label_scale']='100.0'
    rtn['workers']='0'
    rtn['checkpoint_file']=''
    rtn['apply_normalization']='True'


    config_file = open(filepath)
    lines = config_file.readlines()
    vals = []
    for line in lines:
        key, value = line.split(",")
        key = key.replace("\n","")
        value = value.replace("\n","")
        rtn[key]=value
    return rtn
def main():
    parser = argparse.ArgumentParser(description="Steering prediction with PilotNet")
    parser.add_argument("--config_file", type=str, required=True, help="Config file to use")
    args = parser.parse_args()
    config_fp = args.config_file
    config = load_config(config_fp)
    #mandatory parameters
    learning_rate = float(config['learning_rate'])
    root_dir, annotation_file = os.path.split(config['annotation_file'])
    prefix, _ = annotation_file.split(".")

    #optional parameters
    file_prefix = config['file_prefix']
    checkpoint_file = config['checkpoint_file']

    load_files = bool(config['load_files'])
    use_float32 = bool(config['use_float32'])
    apply_norm = bool(config['apply_normalization'])


    label_scale = float(config['label_scale'])
    momentum = float(config['momentum'])

    batch_size = int(config['batch_size'])
    gpu = int(config['gpu'])
    epochs = int(config['epochs'])
    workers = int(config['workers'])

    
    

    _, config_file = os.path.split(config_fp)
    config_file_name, _ = config_file.split(".")
    output_dir = config_file_name.replace("\n","")
    prefix = prefix + file_prefix
    network = models.EnsignNet()
    print(network)
    size=(66,200)
    if(label_scale == 1.0):
        label_transformation = None
    else:
        label_transformation = transforms.Compose([transforms.Lambda(lambda inputs: inputs.mul(label_scale))])
    if(use_float32):
        network.float()
        trainset = loaders.F1Dataset(root_dir, annotation_file, size, use_float32=True, label_transformation = label_transformation)
    else:
        network.double()
        trainset = loaders.F1Dataset(root_dir, annotation_file, size, label_transformation = label_transformation)
    if(gpu>=0):
        network = network.cuda(gpu)
    
   # trainset.read_files()
    
    if(load_files or (not os.path.isfile("./" + prefix+"_images.pkl")) or (not os.path.isfile("./" + prefix+"_annotations.pkl"))):
        trainset.read_files()
        trainset.write_pickles(prefix+"_images.pkl",prefix+"_annotations.pkl")
    else:  
        trainset.read_pickles(prefix+"_images.pkl",prefix+"_annotations.pkl")
    ''' '''
    if apply_norm:
        mean,stdev = trainset.statistics()

        mean_ = torch.from_numpy(mean).float()

        stdev_ = torch.from_numpy(stdev).float()

        print("Mean")
        print(mean_)
        print("Stdev")
        print(stdev_)

        trainset.img_transformation = transforms.Normalize(mean_,stdev_)
    else:
        print("Skipping Normalize")
        trainset.img_transformation = None
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 0)
    print(trainLoader)
    #Definition of our loss.
    criterion = nn.MSELoss()

    # Definition of optimization strategy.
    optimizer = optim.SGD(network.parameters(), lr = learning_rate, momentum=momentum)
    config['image_transformation']=trainset.img_transformation
    if(not os.path.isdir(output_dir)):
        os.makedirs(output_dir)
    config_dump = open(os.path.join(output_dir,"config.pkl"), 'wb')
    pickle.dump(config,config_dump)
    config_dump.close()
    losses = train_model(network, criterion, optimizer, trainLoader, prefix, output_dir, n_epochs = epochs, use_gpu = gpu)
    if(optical_flow):
        loss_path = os.path.join(output_dir,""+prefix+"_"+rnn_cell_type+"_OF.txt")
    else:
        loss_path = os.path.join(output_dir,""+prefix+"_"+rnn_cell_type+".txt")
    f = open(loss_path, "w")
    f.write("\n".join(map(lambda x: str(x), losses)))
    f.close()

if __name__ == '__main__':
    main()