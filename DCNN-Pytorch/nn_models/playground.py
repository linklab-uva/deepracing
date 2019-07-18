import torch
import nn_models.LossFunctions as loss_functions
import nn_models.Models as models
import numpy as np
import torch.optim as optim
import data_loading.backend
import os
import skimage
import skimage.io
db = data_loading.backend.LMDBWrapper()
img_folder = "C:/Users/ttw2x/Documents/f1_data/toy_dataset/images"
db.readDatabase(os.path.join(img_folder,"lmdb"))
im = db.getImage("image_11.jpg")
print(im.shape)
skimage.io.imshow(im)
skimage.io.show()
