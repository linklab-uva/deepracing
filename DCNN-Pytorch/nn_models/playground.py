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
    dataset1 = loaders.F1OpticalFlowDataset("/home/ttw2xk/f1data/test_dataset/linear_1.csv",(66,200),\
      context_length=5, sequence_length=3)
    dataset1.loadFiles()
    #dataset1.loadPickles()
    #dataset1.writePickles()

    image, labels = dataset1[5]


    #dataset2= loaders.F1ImageDataset("/home/ttw2xk/f1data/test_dataset/linear_2.csv",(66,200))
    dataset2 = loaders.F1OpticalFlowDataset("/home/ttw2xk/f1data/test_dataset/linear_2.csv",(66,200),\
      context_length=5, sequence_length=3)
    dataset2.loadFiles()
    #dataset2.loadPickles()
    #dataset2.writePickles()
    image, labels = dataset2[5]

    bigdataset = torch.utils.data.ConcatDataset((dataset1,dataset2))

    print(len(bigdataset))
    image, labels = bigdataset[40]

    indices = np.linspace(0,9,9,endpoint=False).astype(np.int32)

    subset = Subset(bigdataset, indices)


    image, labels = subset[0]

    print(image.shape)
    print(labels.shape)

    print(flows[0])
#    print(flows[24])


    #dataset.writePickles()

if __name__ == '__main__':
    main()
