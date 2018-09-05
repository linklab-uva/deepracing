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
import glob
import argparse
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def train_model(network, criterion, optimizer, load_files,cell_type, file_prefix,context_length,sequence_length,batch_size,label_transformation,workers,epoch,output_dir,use_float32,root_dir, annotation_file, gpu = -1):
           
    cum_loss = 0.0
    num_samples=0
    entry_flag=True

    for file in load_files:
        if(entry_flag and epoch!=0):
            log_path = os.path.join(output_dir,"epoch"+str(epoch)+".model")
            entry_flag=False
        else:
            log_path = os.path.join(output_dir,"epoch"+str(epoch+1)+".model")
        state_dict = torch.load(log_path)
        network.load_state_dict(state_dict)
        #Load partitioned Trainset
        dir,file = file.split('\\')
        prefix,data_type,op,suffix = file.split('_')
        data_type='labels'
        label_file = prefix+'_'+data_type+'_'+op+'_'+suffix
            
        #print('Reading Trainset %s'%(file))
        if(use_float32):
            network.float()
            trainset = loaders.F1SequenceDataset(root_dir,annotation_file,(66,200),\
            context_length=context_length, sequence_length=sequence_length, use_float32=True, label_transformation = label_transformation)
        else:
            network.double()
            trainset = loaders.F1SequenceDataset(root_dir, annotation_file,(66,200),\
            context_length=context_length, sequence_length=sequence_length, label_transformation = label_transformation)
        trainset.read_pickles(os.path.join(dir,file),os.path.join(dir,label_file))
        mean,stdev = trainset.statistics()
        mean_ = torch.from_numpy(mean)
        stdev_ = torch.from_numpy(stdev)
        if use_float32:
            mean_.float()
            stdev_.float()
        trainset.img_transformation = transforms.Normalize(mean_,stdev_)
        trainLoader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = False, num_workers = workers)
        t = tqdm(enumerate(trainLoader),desc='\tTraining Data (Epoch:%d(%d), lr=%f)'%(epoch+1,int(suffix.split('.')[0]),optimizer.param_groups[0]['lr']),leave=True)
        for (i, (inputs, throttle, brake,_, labels,flag)) in t:
            if(all(flag.numpy())):
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
            else:
                continue #using break breaks the tqdm, ignore the iter count.
        log_path = os.path.join(output_dir,"epoch"+str(epoch+1)+".model")
        torch.save(network.state_dict(), log_path)
    return cum_loss,num_samples,optimizer.param_groups[0]['lr']
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
    rtn['hidden_dim']='50'
    rtn['checkpoint_file']=''
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
    super_convergence = bool(config['super_convergence'])

    label_scale = float(config['label_scale'])
    momentum = float(config['momentum'])

    batch_size = int(config['batch_size'])
    gpu = int(config['gpu'])
    epochs = int(config['epochs'])
    workers = int(config['workers']) #Number of workers does not work in windows
    context_length = int(config['context_length'])
    sequence_length = int(config['sequence_length'])
    hidden_dim = int(config['hidden_dim'])

    _, config_file = os.path.split(config_fp)
    config_file_name, _ = config_file.split(".")
    output_dir = config_file_name.replace("\n","")
    
    prefix = prefix + file_prefix

    #Declare the Network
    rnn_cell_type = 'lstm'
    output_dir = output_dir +"_"+rnn_cell_type+'_'+str(learning_rate)
    network = models.AdmiralNet(cell=rnn_cell_type, context_length = context_length, sequence_length=sequence_length, hidden_dim = hidden_dim, use_float32 = use_float32, gpu = gpu)
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
        context_length=context_length, sequence_length=sequence_length, use_float32=True, label_transformation = label_transformation)
    else:
        network.double()
        trainset = loaders.F1SequenceDataset(root_dir, annotation_file,(66,200),\
        context_length=context_length, sequence_length=sequence_length, label_transformation = label_transformation)
    if(gpu>=0):
        network = network.cuda(gpu)
    
    #Create Model Dump Directory
    if(not os.path.isdir(output_dir)):
        os.makedirs(output_dir)

    #Check for data
    pickle_dir,_ = annotation_file.split('.')
    pickle_dir+='_data'
    if(not os.path.exists(pickle_dir)):
        os.makedirs(pickle_dir)
    load_files = glob.glob(pickle_dir+'\saved_image_opticalflow*.pkl')
    if(len(load_files)==0):
        trainset.read_files_flow()
        load_files = glob.glob(pickle_dir+'\saved_image_opticalflow*.pkl')
 
    load_files.sort()
    print(load_files)
    final_losses = []
    final_lrs = []

    #Save the Config File
    config['image_transformation'] = trainset.img_transformation
    config['label_transformation'] = trainset.label_transformation
    #print("Using configuration: ", config)
    config_dump = open(os.path.join(output_dir,"config.pkl"), 'wb')
    pickle.dump(config,config_dump)
    config_dump.close()
    log_path = os.path.join(output_dir,"epoch"+str(1)+".model")
    torch.save(network.state_dict(), log_path)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(network.parameters(), lr = learning_rate, momentum=momentum)
    if gpu>=0:
        criterion = criterion.cuda(gpu)
    #Begin Training
    print("Beginning Training:")
    #network.train()  # This is important to call before training!
    for epoch in range(starting_epoch,epochs):
        final_loss = 0
        final_lr = 0
        num_samples=0
        learning_rate = optimizer.param_groups[0]['lr']
        print("#Epoch %d of %d, lr= %f" %(epoch+1, epochs,optimizer.param_groups[0]['lr']))
                  
        loss,n_samples, lr = train_model(network, criterion, optimizer,load_files,rnn_cell_type, prefix,context_length,sequence_length,batch_size,label_transformation,workers,epoch,output_dir,use_float32,root_dir, annotation_file, gpu = gpu)
        final_loss += loss
        num_samples+=n_samples
        final_lr = lr

        if(super_convergence):
            if(epoch%5!=0):
                optimizer.param_groups[0]['lr'] += learning_rate
            else:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*2
                learning_rate = optimizer.param_groups[0]['lr']
        final_losses.append(final_loss/float(num_samples))
        #print('\nFinal Loss: %f'%(final_loss/float(num_samples)))
        final_lrs.append(final_lr)
       
    #Log Loss progress
    loss_path = os.path.join(output_dir,""+prefix+"_"+rnn_cell_type+"_OF.txt")
    f = open(loss_path, "w")
    f.write("\n".join(map(lambda x: str(x), final_losses)))
    f.close()

    #Save Superconvergence Graph
    if(super_convergence):
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_xscale('log')
        ax.plot(final_lrs,final_losses,'b')
        plt.savefig(os.path.join(output_dir,"admiralnet_super_convergence_plot.jpeg"))
        plt.show()

if __name__ == '__main__':
    main()