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
def run_epoch(network, optimizer, trainLoader, gpu, position_loss, rotation_loss, loss_weights=[1.0, 1.0], imsize=(66,200)):
    cum_loss = 0.0
    batch_size = trainLoader.batch_size
    num_samples=0
    t = tqdm(enumerate(trainLoader))
    network.train()  # This is important to call before training!
    for (i, (image_torch, position_torch, rotation_torch, linear_velocity_torch, angular_velocity_torch, session_time)) in t:
        if gpu>=0:
            image_torch = image_torch.cuda(gpu)
            position_torch = position_torch.cuda(gpu)
            rotation_torch = rotation_torch.cuda(gpu)
      #  print(image_torch.dtype)
        # Forward pass:
        position_predictions, rotation_predictions = network(image_torch)
        #print("Output shape: ", outputs.shape)
        #print("Label shape: ", labels.shape)
        loss = loss_weights[0]*position_loss(position_predictions, position_torch) + loss_weights[1]*rotation_loss(rotation_predictions, rotation_torch)


        # Backward pass:
        optimizer.zero_grad()
        loss.backward() 

        # Weight and bias updates.
        optimizer.step()

        # logging information
        loss_ = loss.item()
        cum_loss += loss_
        num_samples += batch_size
        t.set_postfix(cum_loss = cum_loss/num_samples)

parser = argparse.ArgumentParser(description="Train AdmiralNet Pose Predictor")
parser.add_argument("config_file", type=str,  help="Configuration file to load")
args = parser.parse_args()
config_file = args.config_file
with open(config_file) as f:
    config = yaml.load(f, Loader = yaml.SafeLoader)
dataset_dir = config["dataset_dir"]
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
output_directory = config["output_directory"]
num_workers = config["num_workers"]
image_directory = config.get('image_directory', None)
if os.path.isdir(output_directory):
    s = ""
    while(not (s=="y" or s=="n")):
         s = input("Directory " + output_directory + " already exists. Overwrite it with new data? [y\\n]\n")
    if s=="n":
        print("Thanks for playing!")
        exit(0)
    shutil.rmtree(output_directory)
os.makedirs(output_directory)
net = models.AdmiralNetPosePredictor(gpu=gpu,context_length = context_length, sequence_length = sequence_length,\
    hidden_dim=hidden_dimension, input_channels=input_channels, temporal_conv_feature_factor = temporal_conv_feature_factor)
position_loss = torch.nn.MSELoss(reduction='sum')
rotation_loss = loss_functions.QuaternionDistance()
optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum=momentum)
if gpu>=0:
    rotation_loss = rotation_loss.cuda(gpu)
    position_loss = position_loss.cuda(gpu)
    net = net.cuda(gpu)
dset = data_loading.proto_datasets.ProtoDirDataset(dataset_dir, context_length, sequence_length, image_directory = image_directory)
dataloader = data_utils.DataLoader(dset, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)
yaml.dump(config, stream=open(os.path.join(output_directory,"config.yaml"), "w"), Dumper = yaml.SafeDumper)
for i in range(num_epochs):
    postfix = i + 1
    run_epoch(net, optimizer, dataloader, gpu, position_loss, rotation_loss, loss_weights=loss_weights)
    modelout = os.path.join(output_directory,"epoch_" + str(postfix) + ".model")
    torch.save(net.state_dict(), modelout)