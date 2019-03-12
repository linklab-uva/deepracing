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
import deepf1_image_reading as imreading
def main():
      data_dir = os.path.join('/home/ttw2xk','deepf1data','australia_fullview_run2')
      data_file = 'linear'
      context_length = 10
      sequence_length = 1
      batch_size = 8
   # backend = image_backends.DeepF1ImageTensorBackend(image_tensor=torch.load(os.path.join(data_dir,'linear_image_tensor.pt')), label_tensor=torch.load(os.path.join(data_dir,'linear_label_tensor.pt')))  # backend = image_backends.DeepF1ImageTensorBackend()
    #backend=image_backends.DeepF1ImageTensorBackend(context_length, sequence_length)
    #backend.loadImages(os.path.join(data_dir,data_file+'.csv'),(66,200))
    #torch.save(backend.image_tensor, os.path.join(data_dir,'linear_image_tensor.pt'))
    #torch.save(backend.label_tensor, os.path.join(data_dir,'linear_label_tensor.pt'))
      backend = image_backends.DeepF1LMDBBackend(os.path.join(data_dir,data_file+'.csv'),context_length,sequence_length, imsize = (66,200))
     # backend.readImages(os.path.join(data_dir,'lmdb',data_file+'.mdb'))
      #backend.readDatabase(os.path.join(data_dir,'lmdb',data_file+'.mdb'))
      ofbackend = of_backends.DeepF1LMDBOptFlowBackend(os.path.join(data_dir,data_file+'.csv'),context_length,sequence_length, imsize = (66,200))
     # ofbackend.readImages(os.path.join(data_dir,'lmdb',data_file+'.mdb'))
      ofbackend.readDatabase(os.path.join(data_dir,'lmdb',data_file+'.mdb'))
      #flows = ofbackend.getFlowsRange(0)
      #print(flows.shape)
    #
      # images = imreading.readImages(os.path.join(data_dir,'raw_images','raw_image_'), 1, context_length, cv2.IMREAD_UNCHANGED)
      # print(images[0])
      # print(len(images))

      ofds = loaders.F1OpticalFlowDataset(ofbackend)
      ds = loaders.F1ImageSequenceDataset(backend)
      # images,labels=ds[0]
      # print(images.shape)
      # print(images.type())
   

      trainLoader = torch.utils.data.DataLoader(ofds, batch_size = batch_size, shuffle = True, num_workers = 6, drop_last=True )
      t = tqdm(enumerate(trainLoader), leave=True)
      cv2.namedWindow("imnp0", cv2.WINDOW_AUTOSIZE)
      for (i, (inputs, labels)) in t:
        # print(inputs.shape)
         #print(labels.shape)
         #imtorch = inputs[batch_size-1][5]
         #imnp = cv2.cvtColor(np.round(255.0*imtorch.numpy().transpose(1,2,0)).astype(np.uint8), cv2.COLOR_RGB2BGR)
         #cv2.imshow("imnp",imnp)
         #cv2.waitKey(100)
         pass



  
if __name__ == '__main__':
  main()
