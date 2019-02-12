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
def main():
  data_dir='E:/deepf1data/australia_fullview_run2'
  backend = image_backends.DeepF1ImageTensorBackend(image_tensor=torch.load(os.path.join(data_dir,'linear_image_tensor.pt')), label_tensor=torch.load(os.path.join(data_dir,'linear_label_tensor.pt')))


 # backend=image_backends.DeepF1ImageTensorBackend()
 # backend.loadImages(os.path.join(data_dir,'linear.csv'),(66,200))
 # torch.save(backend.image_tensor, os.path.join(data_dir,'linear_image_tensor.pt'))
 # torch.save(backend.label_tensor, os.path.join(data_dir,'linear_label_tensor.pt'))


  ds = loaders.F1ImageSequenceDataset(backend)
  images, labels = ds[500]
  print(images.shape)
  print(labels.shape)
  
  flow_ds = loaders.F1OpticalFlowDataset(backend)
  flows, labels = flow_ds[len(flow_ds)-1]
  print(flows.shape)
  print(flows.shape)
if __name__ == '__main__':
  main()
