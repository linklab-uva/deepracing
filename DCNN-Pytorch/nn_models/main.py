import torch
import cv2
import numpy as np
import nn_models
import data_loading.image_loading as il
import nn_models.Models as models
im = il.load_image("test_image.jpg")
print(im)
net = models.PilotNet()
print(net)