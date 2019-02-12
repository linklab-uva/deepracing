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
  image_tensor = torch.load('/home/ttw2xk/deepf1data/australia_fullview_run2/linear_image_tensor.pt')
  label_tensor = torch.load('/home/ttw2xk/deepf1data/australia_fullview_run2/linear_label_tensor.pt')
  backend=image_backends.DeepF1ImageTensorBackend(image_tensor=image_tensor, label_tensor=label_tensor)

 # backend.loadImages('/home/ttw2xk/deepf1data/australia_fullview_run2/linear.csv',(66,200))
 # torch.save(backend.image_tensor, '/home/ttw2xk/deepf1data/australia_fullview_run2/linear_image_tensor.pt')
  #torch.save(backend.label_tensor, '/home/ttw2xk/deepf1data/australia_fullview_run2/linear_label_tensor.pt')

  ds = loaders.F1ImageSequenceDataset(backend)
  images, labels = ds[0]
  print(images.shape)
  print(labels.shape)
  flows = loaders.imagesToFlow(images)
  print(flows.shape)
if __name__ == '__main__':
  main()
