import torch
import cv2
import numpy as np
import nn_models
import data_loading.image_loading as il
import nn_models.Models as models
import numpy.random
im = il.load_image("test_image.jpg",size=(66,200),scale_factor=255.0)
print(im)
pilot_net = models.PilotNet()
pilot_net.double()
pilot_net.train()
print(pilot_net)
input = torch.from_numpy(np.random.rand(10,3,66,200))
results = pilot_net(input)
print(results)


captain_net = models.CaptainNet()
captain_net.double()
captain_net.train()
print(captain_net)
input = torch.from_numpy(np.random.rand(10,3,66,200))
results = captain_net(input)
print(results)