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
        # Loss is a variable, and calling backward on a Variable will
        # compute all the gradients that lead to that Variable taking on its
        # current value.
        loss.backward() 

        # Weight and bias updates.
        optimizer.step()

        # logging information.
        cum_loss += loss.item()
        num_samples += batch_size
        t.set_postfix(cum_loss = cum_loss/num_samples)
 

def train_model(network, criterion, optimizer, trainLoader, file_prefix, n_epochs = 10, use_gpu = False):
    if use_gpu:
        network = network.cuda()
        criterion = criterion.cuda()
    # Training loop.
    for epoch in range(0, n_epochs):
        print("Epoch %d of %d" %((epoch+1),n_epochs))
        run_epoch(network, criterion, optimizer, trainLoader, use_gpu)
        log_path = os.path.join("log",""+file_prefix+str((epoch+1))+ ".model")
        torch.save(network.state_dict(), log_path)
im = il.load_image("test_image.jpg",size=(66,200),scale_factor=255.0)
print(im)
network = models.PilotNet()
network.float()

learningRate = 0.01
trainset = loaders.F1Dataset("D:\\test_data\\slow_australia_track_run3","run3_linear.csv",(3,66,200),1)
print(trainset)
trainset.read_pickles('slow_australia_track_run3_images.pkl','slow_australia_track_run3_linear.pkl')
trainLoader = torch.utils.data.DataLoader(trainset, batch_size = 8, shuffle = True, num_workers = 0)
print(trainLoader)
#Definition of our loss.
criterion = nn.MSELoss()

# Definition of optimization strategy.
optimizer = optim.SGD(network.parameters(), lr = learningRate, momentum=0.01)

train_model(network, criterion, optimizer, trainLoader, "pilotnet_linear_interpolation", n_epochs = 200, use_gpu = True)