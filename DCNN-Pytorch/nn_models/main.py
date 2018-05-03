import torch
import cv2
import numpy as np
import nn_models
import data_loading.image_loading as il
import nn_models.Models as models
import numpy.random
im = il.load_image("test_image.jpg",size=(66,200),scale_factor=255.0)
print(im)
net = models.PilotNet()
net.double()
net.eval()
print(net)
input = torch.from_numpy(np.random.rand(1,3,66,200))
input[0]=torch.from_numpy(im)
results = net(input)
print(results)