from sklearn.decomposition import PCA
import numpy as np
import os
import argparse
import skimage
import skimage.io
import shutil
import deepracing.imutils
import deepracing.backend
import cv2
import deepracing.imutils
import random
import yaml
import torch
import torch.utils.data as data_utils
import data_loading.proto_datasets
import deepracing.backend
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
from tqdm import tqdm as tqdm
import os
import pickle
parser = argparse.ArgumentParser(description="Display Scree Plot of  a dataset")
parser.add_argument("dataset_config_file", type=str, help="Dataset config file to load data from")
#parser.add_argument("--display_resize_factor", type=float, default=0.5, help="Resize the first image by this factor for selecting a ROI.")
args = parser.parse_args()
main(args)
print("Hello World!")
dataset_config_file = args.dataset_config_file
dataset_name = os.path.splitext(os.path.basename(dataset_config_file))[0]
with open(dataset_config_file) as f:
    dataset_config = yaml.load(f, Loader = yaml.SafeLoader)
print(dataset_config)
max_spare_txns = 16
image_size = dataset_config["image_size"]
datasets = dataset_config["datasets"]
dsets=[]
numsamples = 0
for dataset in datasets:
    print("Parsing database config: %s" %(str(dataset)))
    image_folder = dataset["image_folder"]
    image_lmdb = os.path.join(image_folder,"image_lmdb")
    label_folder = dataset["label_folder"]
    label_lmdb = os.path.join(label_folder,"lmdb")
    key_file = dataset["key_file"]

    label_wrapper = deepracing.backend.ControlLabelLMDBWrapper()
    label_wrapper.readDatabase(label_lmdb, mapsize=3e9, max_spare_txns=max_spare_txns )

    image_size = np.array(image_size)
    image_mapsize = float(np.prod(image_size)*3+12)*float(len(label_wrapper.getKeys()))*1.1
    image_wrapper = deepracing.backend.ImageLMDBWrapper(direct_caching=False)
    image_wrapper.readDatabase(image_lmdb, max_spare_txns=max_spare_txns, mapsize=image_mapsize )
    
    curent_dset = data_loading.proto_datasets.ControlOutputDataset(image_wrapper, label_wrapper, key_file, image_size = image_size)
    dsets.append(curent_dset)
    numsamples+=len(curent_dset)
if len(dsets) == 0:
    print("No datasets to process. Exiting")
    exit(0)
elif len(dsets) == 1:   
    dset = dsets[0]
else:
    dset = data_utils.ConcatDataset(dsets)
exampleim = dset[0][0]
numfeatures = exampleim.numel()
N = numsamples
# N = 50
dataloader = data_utils.DataLoader(dset, batch_size=N, shuffle=True)
print("Loading data")
batch_images, batch_labels = next(iter(dataloader))
print("Loaded data")
datamatrix_torch = batch_images.reshape(N,numfeatures).float()
datamatrix = datamatrix_torch.numpy()
# U,S,V = torch.svd(datamatrix_torch_cuda)
# print("Got SVD")
pca = PCA()
pca.fit(datamatrix)
variance_ratios = pca.explained_variance_ratio_
I = np.linspace(1,variance_ratios.shape[0],variance_ratios.shape[0]).astype(np.int32)
fig = plt.figure("Scree Plot")
plt.plot(I, variance_ratios, label='Ratio of Explained Variance')
fig.legend()
plt.savefig('scree_plot.png')
with open(dataset_name+"_pca.pkl",'wb') as f:
    pickle.dump(pca,f)