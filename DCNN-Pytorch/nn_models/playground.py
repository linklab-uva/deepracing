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
import data_loading.backend.ImageBackend as image_backends
import data_loading.backend.OpticalFlowBackend as of_backends
from tqdm import tqdm
import pickle as pkl
def main():
    data_dir='/home/ttw2xk/deepf1data/australia_fullview_run2'
    backend = image_backends.DeepF1ImageTensorBackend(image_tensor=torch.load(os.path.join(data_dir,'linear_image_tensor.pt')), label_tensor=torch.load(os.path.join(data_dir,'linear_label_tensor.pt')))
  #  # backend = image_backends.DeepF1ImageTensorBackend()
   # backend.loadImages(os.path.join(data_dir,'linear.csv'),(66,200))
   # torch.save(backend.image_tensor, os.path.join(data_dir,'linear_image_tensor.pt'))
   # torch.save(backend.label_tensor, os.path.join(data_dir,'linear_label_tensor.pt'))
    # numpybackend = image_backends.DeepF1NumpyArrayBackend()
    # numpybackend.loadImages(os.path.join(data_dir,'linear.csv'),(66,200))
    # pkl.dump(numpybackend.image_array, open(os.path.join(data_dir,'linear_numpy_image_array.pkl'), 'wb'))
    # pkl.dump(numpybackend.label_array, open(os.path.join(data_dir,'linear_numpy_label_array.pkl'), 'wb'))
    # numpybackend = image_backends.DeepF1NumpyArrayBackend(image_array=pkl.load(open(os.path.join(data_dir,'linear_numpy_image_array.pkl'), 'rb')),\
    #                                                       label_array=pkl.load(open(os.path.join(data_dir,'linear_numpy_label_array.pkl'), 'rb')))
    
    # of_backend = of_backends.DeepF1OpticalFlowTensorBackend(flow_tensor=torch.load(os.path.join(data_dir,'linear_optflow_tensor.pt')), label_tensor=torch.load(os.path.join(data_dir,'linear_optflow_label_tensor.pt')))
    # print(of_backend.flow_tensor.shape)
    # print(of_backend.label_tensor.shape)
    #of_backend = of_backends.DeepF1OpticalFlowTensorBackend()
    #of_backend.loadFlows( os.path.join(data_dir,'linear.csv') )
    #torch.save(of_backend.flow_tensor, os.path.join(data_dir,'linear_optflow_tensor.pt'))
    #torch.save(of_backend.label_tensor, os.path.join(data_dir,'linear_optflow_label_tensor.pt'))

    # backend=image_backends.DeepF1ImageTensorBackend()
    


    ds = loaders.F1ImageSequenceDataset(backend)
    images, labels = ds[500]
  
    flow_ds = loaders.F1OpticalFlowDataset(backend)
    print(ds.backend)
    print(flow_ds.backend)



    trainLoader = torch.utils.data.DataLoader(flow_ds, batch_size = 8, shuffle = True, num_workers = 8)
    t = tqdm(enumerate(trainLoader), leave=True)
    for (i, (inputs, labels)) in t:
        pass
        #print(inputs.shape)
        #print(labels.shape)


  
if __name__ == '__main__':
  main()
