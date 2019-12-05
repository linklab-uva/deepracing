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
import numpy.linalg as la
parser = argparse.ArgumentParser(description="Display Scree Plot of  a dataset")
parser.add_argument("dataset_config_file", type=str, help="Dataset config file to load data from")
parser.add_argument("--use_sklearn", action="store_true", help="Attempt to use ScikitLearn's built-in PCA method")
#parser.add_argument("--display_resize_factor", type=float, default=0.5, help="Resize the first image by this factor for selecting a ROI.")
args = parser.parse_args()
print("Hello World!")
use_sklearn = args.use_sklearn
dataset_config_file = args.dataset_config_file
dataset_name = os.path.splitext(os.path.basename(dataset_config_file))[0]
with open(dataset_config_file) as f:
    dataset_config = yaml.load(f, Loader = yaml.SafeLoader)
print(dataset_config)
max_spare_txns = 16
image_size = np.round((np.array(dataset_config["image_size"], dtype=np.float64)/1.0)).astype(np.int32)
datasets = dataset_config["datasets"]

# N = 50
matrix_file = dataset_name+"_data_matrix.pt"
if os.path.isfile(matrix_file):
    print("Using data matrix in %s" %(matrix_file))
    with open(matrix_file,'rb') as f:
        datamatrix_torch = torch.load(f).float()
    N = datamatrix_torch.shape[0]
else:
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
    dataloader = data_utils.DataLoader(dset, batch_size=N, shuffle=True)
    print("Loading data")
    batch_images, _ = next(iter(dataloader))
    print("Loaded data")
    datamatrix_torch = batch_images.reshape(N,numfeatures).float()
    with open(matrix_file,'wb') as f:
        torch.save(datamatrix_torch,f)
    del dataloader
    del dset
    del image_wrapper
    del label_wrapper
# pca = PCA()
# print("Fitting PCA to data")
# pca.fit(datamatrix_torch.numpy())
# variance_ratios = pca.explained_variance_ratio_
# datamatrix_torch_cuda = datamatrix_centered.cuda(0)


covariance_file = dataset_name+"_covariance.pt"
if os.path.isfile(covariance_file):
    print("Using covariance matrix in %s" %(covariance_file))
    with open(covariance_file,'rb') as f:
        C = torch.load(f).float()
else:
    print("Centering data")
    datamatrix_centered = torch.sub( datamatrix_torch, torch.mean(datamatrix_torch,0) ) 
    datamatrix_centered_np = datamatrix_centered.numpy()#.copy()
    print("Computing Covariance Matrix")
    #C = (1/(N-1))*torch.matmul(datamatrix_torch_cuda.transpose(0,1), datamatrix_torch_cuda)
    # C_np = C.cpu().numpy().copy()
    C = (1/(N-1))*torch.matmul(datamatrix_centered.transpose(0,1), datamatrix_centered)
    with open(covariance_file,'wb') as f:
        torch.save(C.cpu(),f)
C_cuda = C.cuda(0)
C_np = C.numpy()#.copy()
del datamatrix_torch
print("Shape of covariance matrix: " + str(C.shape))
print("Doing Eigenvalue Decomposition")
eigenvalues_real, eigenvectors = torch.symeig(C, eigenvectors=True)
variances = torch.flip(eigenvalues_real,(0,))
eigenvectors_sorted = torch.flip(eigenvectors,(1,))
with open(dataset_name+"_eigenvalues.pt",'wb') as f:
    torch.save(variances,f)
with open(dataset_name+"_eigenvectors.pt",'wb') as f:
    torch.save(eigenvectors_sorted,f)

variance_ratios = (variances/torch.sum(variances)).numpy()
I = np.linspace(1,variance_ratios.shape[0],variance_ratios.shape[0]).astype(np.int32)
fig = plt.figure("Scree Plot")
plt.plot(I, variance_ratios, label='Variance Ratios')
fig.legend()
plt.savefig('scree_plot.png')
