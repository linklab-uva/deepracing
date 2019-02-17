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
import data_loading.backend.ImageSequenceBackend as image_backends
import data_loading.backend.OpticalFlowBackend as of_backends
from tqdm import tqdm
import pickle as pkl
import time
def main():
    data_dir = os.path.join('E:\\','deepf1data','australia_fullview_run2')
    data_file = 'linear'
    context_length = 10
    sequence_length = 1
   # backend = image_backends.DeepF1ImageTensorBackend(image_tensor=torch.load(os.path.join(data_dir,'linear_image_tensor.pt')), label_tensor=torch.load(os.path.join(data_dir,'linear_label_tensor.pt')))  # backend = image_backends.DeepF1ImageTensorBackend()
    #backend=image_backends.DeepF1ImageTensorBackend(context_length, sequence_length)
    #backend.loadImages(os.path.join(data_dir,data_file+'.csv'),(66,200))
    #torch.save(backend.image_tensor, os.path.join(data_dir,'linear_image_tensor.pt'))
    #torch.save(backend.label_tensor, os.path.join(data_dir,'linear_label_tensor.pt'))
    backend = image_backends.DeepF1ImageDirectoryBackend(os.path.join(data_dir,data_file+'.csv'),context_length,sequence_length, imsize = (66,200))
    #
    


    #ds = loaders.F1ImageSequenceDataset(numpybackend)
    ds = loaders.F1ImageSequenceDataset(backend)
    #images,labels=ds[ds.backend.index_order[0]]
   # print(images)
   

    trainLoader = torch.utils.data.DataLoader(ds, batch_size = 8, shuffle = True, num_workers = 5)
    t = tqdm(enumerate(trainLoader), leave=True)
    for (i, (inputs, labels)) in t:
        pass



  
if __name__ == '__main__':
  main()
