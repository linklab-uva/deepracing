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

dl = loaders.F1OpticalFlowDataset("/zf18/ttw2xk/deepf1data/australia_fullview_run2/linear.csv",(66,200), 25, 10)
dl.loadFiles()