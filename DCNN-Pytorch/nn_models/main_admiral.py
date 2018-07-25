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
import matplotlib.pyplot as plt

def run_epoch(network, criterion, optimizer, trainLoader, gpu = -1):
    network.train()  # This is important to call before training!
    cum_loss = 0.0
    batch_size = trainLoader.batch_size
    num_samples=0
    t = tqdm(enumerate(trainLoader))
    for (i, (inputs, throttle, brake,_, labels)) in t:
        if gpu>=0:
            inputs = inputs.cuda(gpu)
            throttle = throttle.cuda(gpu)
            brake= brake.cuda(gpu)
            labels = labels.cuda(gpu)
        # Forward pass:
        outputs = network(inputs,throttle,brake)
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
    return cum_loss/num_samples
 

def train_model(network, criterion, optimizer, trainLoader,cell_type, file_prefix, directory,super_convergence, n_epochs = 10, gpu = -1, starting_epoch = 0):
    if gpu>=0:
        criterion = criterion.cuda(gpu)
    # Training loop.
    if(not os.path.isdir(directory)):
        os.makedirs(directory)
    losses = []
    lrs = []
    learning_rate = optimizer.param_groups[0]['lr']
    for epoch in range(starting_epoch, starting_epoch + n_epochs):
        epoch_num = epoch + 1
        print("Epoch %d of %d, lr= %f" %(epoch_num, n_epochs,optimizer.param_groups[0]['lr']))
        loss = run_epoch(network, criterion, optimizer, trainLoader, gpu)
        losses.append(loss)
        lrs.append(optimizer.param_groups[0]['lr'])
        if(super_convergence):
            if(epoch_num%5!=0):
                optimizer.param_groups[0]['lr'] += learning_rate
            else:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*2
                learning_rate = optimizer.param_groups[0]['lr']
        log_path = os.path.join(directory,""+file_prefix+"_epoch"+str(epoch_num)+".model")
        torch.save(network.state_dict(), log_path)
    return losses, lrs
def load_config(filepath):
    rtn = dict()
    rtn['batch_size']='8'
    rtn['gpu']='-1'
    rtn['epochs']='10'
    rtn['momentum']='0.0'
    rtn['file_prefix']=''
    rtn['load_files']=''
    rtn['use_float32']=''
    rtn['label_scale']='100.0'
    rtn['workers']='0'
    rtn['context_length']='10'
    rtn['sequence_length']='10'
    rtn['hidden_dim']='100'
    rtn['checkpoint_file']=''
    rtn['optical_flow']=''
    rtn['super_convergence']=''


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
    optical_flow = bool(config['optical_flow'])
    super_convergence = bool(config['super_convergence'])

    label_scale = float(config['label_scale'])
    momentum = float(config['momentum'])

    batch_size = int(config['batch_size'])
    gpu = int(config['gpu'])
    epochs = int(config['epochs'])
    workers = int(config['workers'])
    context_length = int(config['context_length'])
    sequence_length = int(config['sequence_length'])
    hidden_dim = int(config['hidden_dim'])
    

    
    

    _, config_file = os.path.split(config_fp)
    config_file_name, _ = config_file.split(".")
    output_dir = config_file_name.replace("\n","")
    
    prefix = prefix + file_prefix
    rnn_cell_type = 'lstm'
    output_dir = output_dir +"_"+rnn_cell_type
    network = models.AdmiralNet(cell=rnn_cell_type, context_length = context_length, sequence_length=sequence_length, hidden_dim = hidden_dim, use_float32 = use_float32, gpu = gpu, optical_flow = optical_flow)
    starting_epoch = 0
    if(checkpoint_file!=''):
        dir, file = os.path.split(checkpoint_file)
        _,ep = file.split("epoch")
        num, ext = ep.split(".")
        starting_epoch = int(num)
        print("Starting Epoch number:",  starting_epoch)
        state_dict = torch.load(checkpoint_file)
        network.load_state_dict(state_dict)
    if(label_scale == 1.0):
        label_transformation = None
    else:
        label_transformation = transforms.Compose([transforms.Lambda(lambda inputs: inputs.mul(label_scale))])
    if(use_float32):
        network.float()
        trainset = loaders.F1SequenceDataset(root_dir,annotation_file,(66,200),\
        context_length=context_length, sequence_length=sequence_length, use_float32=True, label_transformation = label_transformation, optical_flow = optical_flow)
    else:
        network.double()
        trainset = loaders.F1SequenceDataset(root_dir, annotation_file,(66,200),\
        context_length=context_length, sequence_length=sequence_length, label_transformation = label_transformation, optical_flow = optical_flow)
    if(gpu>=0):
        network = network.cuda(gpu)
    
    
   # trainset.read_files()
    if optical_flow:
        if(load_files or (not os.path.isfile("./" + prefix+"_opticalflows.pkl")) or (not os.path.isfile("./" + prefix+"_opticalflowannotations.pkl"))):
            trainset.read_files_flow()
            trainset.write_pickles(prefix+"_opticalflows.pkl",prefix+"_opticalflowannotations.pkl")
        else:  
            trainset.read_pickles(prefix+"_opticalflows.pkl",prefix+"_opticalflowannotations.pkl")
    else:
        if(load_files or (not os.path.isfile("./" + prefix+"_images.pkl")) or (not os.path.isfile("./" + prefix+"_annotations.pkl"))):
            trainset.read_files()
            trainset.write_pickles(prefix+"_images.pkl",prefix+"_annotations.pkl")
        else:  
            trainset.read_pickles(prefix+"_images.pkl",prefix+"_annotations.pkl")

    mean,stdev = trainset.statistics()
    mean_ = torch.from_numpy(mean).float()
    stdev_ = torch.from_numpy(stdev).float()
    trainset.img_transformation = transforms.Normalize(mean_,stdev_)
   # trainset.img_transformation = transforms.Normalize([2.5374081e-06, -3.1837547e-07] , [3.0699273e-05, 5.9349504e-06])
    print(trainset.img_transformation)
    config['image_transformation'] = trainset.img_transformation
    config['label_transformation'] = trainset.label_transformation
    print("Using configuration: ", config)
    if(not os.path.isdir(output_dir)):
        os.makedirs(output_dir)
    config_dump = open(os.path.join(output_dir,"config.pkl"), 'wb')
    pickle.dump(config,config_dump)
    config_dump.close()
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = workers)
    #Definition of our loss.
    criterion = nn.MSELoss()

    # Definition of optimization strategy.
    optimizer = optim.SGD(network.parameters(), lr = learning_rate, momentum=momentum)
    losses, lrs = train_model(network, criterion, optimizer, trainLoader,rnn_cell_type, prefix, output_dir,super_convergence, n_epochs = epochs, gpu = gpu, starting_epoch = starting_epoch)
    if(optical_flow):
        loss_path = os.path.join(output_dir,""+prefix+"_"+rnn_cell_type+"_OF.txt")
    else:
        loss_path = os.path.join(output_dir,""+prefix+"_"+rnn_cell_type+".txt")
    f = open(loss_path, "w")
    f.write("\n".join(map(lambda x: str(x), losses)))
    f.close()

    if(super_convergence):
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_xscale('log')
        ax.plot(lrs,losses,'b')
        plt.savefig("admiralnet_super_convergence_plot.jpeg")
        plt.show()

if __name__ == '__main__':
    main()