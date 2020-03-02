import comet_ml
import torch
import torch.nn as NN
import torch.utils.data as data_utils
import deepracing_models.data_loading.proto_datasets as PD
from tqdm import tqdm as tqdm
import deepracing_models.nn_models.LossFunctions as loss_functions
import deepracing_models.nn_models.Models
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
import socket
import json
import deepracing_models.data_loading.proto_datasets
from deepracing_models.data_loading.proto_datasets import load_sequence_label_datasets as LD
from torch.utils.tensorboard import SummaryWriter
import deepracing_models.nn_models as nn_models

writer : SummaryWriter = SummaryWriter('D:\\tensorboard\\bezier_predictor')
model_weight_file = 'D:/f1_model_files/bezier_predictor/with_geometric_variants/cleansplinedata_comet/bezier_predictor_no_optflow_no_vel_loss_no_param_loss/epoch_100_params.pt'
model_weight_dir = os.path.dirname(model_weight_file)
dataset_config_file = os.path.join(model_weight_dir, "dataset_config.yaml")
model_config_file = os.path.join(model_weight_dir, "config.yaml")


with open(dataset_config_file, "r") as f:
    dataset_config = yaml.load(f,Loader=yaml.SafeLoader)

    
with open(model_config_file, "r") as f:
    model_config = yaml.load(f,Loader=yaml.SafeLoader)
dsets = LD(dataset_config, model_config)
if len(dsets)==1:
    dset = dsets[0]
else:
    dset = torch.utils.data.ConcatDataset(dsets)
dloader = torch.utils.data.DataLoader(dset,batch_size=16,shuffle=True, num_workers=0, pin_memory=True)
#dataiter = iter(dloader)
images_torch, opt_flow_torch, positions_torch, quats_torch, linear_velocities_torch, angular_velocities_torch, session_times_torch = dset[0]
print(images_torch.shape)
print(positions_torch.shape)
net : nn_models.Models.AdmiralNetCurvePredictor= nn_models.Models.AdmiralNetCurvePredictor(context_length=model_config["context_length"], input_channels=model_config["input_channels"], params_per_dimension=model_config["bezier_order"]+1) 
net.load_state_dict(torch.load(model_weight_file, map_location="cpu"))
net = net.double().cuda(0)
writer.add_images("Some images", images_torch)
writer.add_graph(net, images_torch.double().cuda().unsqueeze(0))
writer.add_graph(net.state_encoder, images_torch.double().cuda())
writer.close()