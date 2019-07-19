import torch
import torch.utils.data as data_utils
import data_loading.proto_datasets
import data_loading.backend
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
def run_epoch(network, optimizer, trainLoader, gpu, position_loss, rotation_loss, loss_weights=[1.0, 1.0], imsize=(66,200), debug=False):
    cum_loss = 0.0
    cum_rotation_loss = 0.0
    cum_position_loss = 0.0
    batch_size = trainLoader.batch_size
    num_samples=0
    t = tqdm(enumerate(trainLoader))
    network.train()  # This is important to call before training!
    for (i, (image_torch, position_torch, rotation_torch, linear_velocity_torch, angular_velocity_torch, session_time)) in t:
        if debug:
            image_np = image_torch[0][0].numpy().copy()
            image_np = skimage.util.img_as_ubyte(image_np.transpose(1,2,0))
            skimage.io.imshow(image_np)
            skimage.io.show()
        if gpu>=0:
            image_torch = image_torch.cuda(gpu)
            position_torch = position_torch.cuda(gpu)
            rotation_torch = rotation_torch.cuda(gpu)
      #  print(image_torch.dtype)
        # Forward pass:
        position_predictions, rotation_predictions = network(image_torch)
        positions_nan = torch.sum(position_predictions!=position_predictions)!=0
        rotation_nan = torch.sum(rotation_predictions!=rotation_predictions)!=0
        if(rotation_nan):
            print(rotation_predictions)
            print("Rotation prediction has a NaN!!!")
            continue
        if(positions_nan):
            print(position_predictions)
            print("Position prediction has a NaN!!!")
            continue
        #print("Output shape: ", outputs.shape)
        #print("Label shape: ", labels.shape)
        rotation_loss_ = rotation_loss(rotation_predictions, rotation_torch)
        position_loss_ = position_loss(position_predictions, position_torch)
        loss = loss_weights[0]*position_loss_ + loss_weights[1]*rotation_loss_


        # Backward pass:
        optimizer.zero_grad()
        loss.backward() 

        # Weight and bias updates.
        optimizer.step()

        # logging information
        cum_loss += loss.item()
        cum_position_loss += position_loss_.item()
        cum_rotation_loss += rotation_loss_.item()
        num_samples += batch_size
        t.set_postfix({"cum_loss" : cum_loss/num_samples, "position_loss" : cum_position_loss/num_samples, "rotation_loss" : cum_rotation_loss/num_samples})

parser = argparse.ArgumentParser(description="Train AdmiralNet Pose Predictor")
parser.add_argument("config_file", type=str,  help="Configuration file to load")
parser.add_argument("--debug", action="store_true",  help="Display images upon each iteration of the training loop")
args = parser.parse_args()
config_file = args.config_file
debug = args.debug
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
image_db_directory = config['image_db_directory']
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
db = data_loading.backend.LMDBWrapper()
db.readDatabase(image_db_directory)
dset = data_loading.proto_datasets.ProtoDirDataset(dataset_dir, context_length, sequence_length, db, image_size = np.array(image_size))
dataloader = data_utils.DataLoader(dset, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)
yaml.dump(config, stream=open(os.path.join(output_directory,"config.yaml"), "w"), Dumper = yaml.SafeDumper)
for i in range(num_epochs):
    postfix = i + 1
    print("Running Epoch Number %d" %(postfix))
    run_epoch(net, optimizer, dataloader, gpu, position_loss, rotation_loss, loss_weights=loss_weights, debug=debug)
    modelout = os.path.join(output_directory,"epoch_%d.model" %(postfix))
    torch.save(net.state_dict(), modelout)