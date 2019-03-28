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
import lmdb

def main():
    #data_dir = os.path.join('E:\\','deepf1data','australia_fullview_run2')
    data_dir = os.path.join('/home','ttw2xk','deepf1data','australia_fullview_run2')
    lmdb_dir = os.path.join(data_dir,'lmdb')
    if not os.path.isdir(lmdb_dir):
        os.mkdir(lmdb_dir)
    data_file = 'linear'
    LMDB_MAP_SIZE=1e9
    f = open(os.path.join(data_dir,data_file+'.csv'))
    annotations = f.readlines()
    f.close()

    LMDB_MAP_SIZE = 1e9 
    env = lmdb.open(os.path.join(lmdb_dir,data_file+'.mdb'), map_size=LMDB_MAP_SIZE)


    #print(">>> Write database...")
    # with env.begin(write=True) as txn:
    #     for (i,line) in tqdm(enumerate(annotations)):
    #         fp, ts, steer, throttle, brake = line.replace("\n","").split(",")
    #         im = cv2.resize(cv2.imread(os.path.join(data_dir,'raw_images',fp), cv2.IMREAD_UNCHANGED), (200,66))#.transpose(2,0,1)
    #         txn.put(fp.encode('ascii'),im.flatten().tostring())
    #       #  txn.commit()


    print(">>> Read database...")
    env = lmdb.open(os.path.join(lmdb_dir,data_file+'.mdb'))#, map_size=1e3)
    random.shuffle(annotations)
    with env.begin(write=False) as txn:
        cv2.namedWindow("dbimage",cv2.WINDOW_AUTOSIZE)
        for (i,line) in tqdm(enumerate(annotations)):
            fp, ts, steer, throttle, brake = line.replace("\n","").split(",")
            im_flat = txn.get(fp.encode('ascii'))
            dbimage = np.frombuffer( im_flat,dtype=np.uint8).reshape(66,200,3)
            cv2.imshow("dbimage",dbimage)
            cv2.waitKey(10)
            
          #  txn.commit()

      
  
if __name__ == '__main__':
  main()
