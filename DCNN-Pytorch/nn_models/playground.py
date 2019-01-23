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
from torch.utils.data.dataset import Subset
def main():
   # parser = argparse.ArgumentParser(description="Playground")
   # parser.add_argument("--dataset_file", type=str, required=True, help="Dataset file to use")
  #  args = parser.parse_args()
    #dataset1 = loaders.F1ImageDataset("/home/ttw2xk/f1data/test_dataset/linear_1.csv",(66,200))
    context_length = 10
    sequence_length = 1
    gpu = 0
    input_channels = 3
    size = (66,200)
    output_dimension = 1
    hidden_dimension = 100


    dataset = loaders.F1CombinedDataset("D:/test_data/australia_fullview_run1/fullview_test.csv",size,\
      context_length=context_length, sequence_length=sequence_length)
    dataset.loadFiles()

    network = models.AdmiralNet_V2(gpu=gpu,context_length = context_length, sequence_length = sequence_length,\
    hidden_dim=hidden_dimension, output_dimension = output_dimension, input_channels=input_channels)
    network = network.cuda(gpu)

    inp = torch.rand(1, context_length, input_channels, size[0], size[1]).cuda(gpu)
    print(inp.shape)

    output = network(inp)

    print(output.shape)

#    print(flows[24])


    #dataset.writePickles()

if __name__ == '__main__':
    main()
