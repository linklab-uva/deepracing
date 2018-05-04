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
network = models.PilotNet()
network.float()
network.cuda()
print(network.state_dict())

modelpath = 'log/run4_linear_epoch200.model'
state_dict = torch.load(modelpath)
network.load_state_dict(state_dict)
print(network.state_dict())

