import torch
import nn_models.LossFunctions as loss_functions
import nn_models.Models as models
import numpy as np
import torch.optim as optim
import data_loading.backend
import os
db = data_loading.backend.LMDBWrapper(np.array((66,200)))
img_folder = "D:\\test_data\\toy_dataset\\images"
keys = [fname for fname in  os.listdir(img_folder) if os.path.isfile(fname) and os.path.splitext(fname)[1]==".jpg"]
print(keys)