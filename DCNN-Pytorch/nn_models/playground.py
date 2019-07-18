import torch
import nn_models.LossFunctions as loss_functions
import nn_models.Models as models
import numpy as np

gpu=0
context_length = np.random.randint(1,high=20)
sequence_length = np.random.randint(1,high=20)
hidden_dimension=100
input_channels = 3
temporal_conv_feature_factor = 2
net = models.AdmiralNetPosePredictor(gpu=gpu,context_length = context_length, sequence_length = sequence_length,\
    hidden_dim=hidden_dimension, input_channels=input_channels, temporal_conv_feature_factor = temporal_conv_feature_factor)
net = net.cuda(gpu)
net.train()
batch_size = 16
inp = torch.rand(batch_size,context_length,input_channels, 66, 200)
inp= inp.cuda(gpu)
loss = loss_functions.QuaternionDistance()
loss = loss.cuda(gpu)

print("Starting")
position_predictions, rotation_predictions = net(inp)
print(position_predictions.shape)
print(rotation_predictions.shape)
rotation_targets = torch.rand(batch_size, sequence_length, 4)
rotation_target_norms = torch.norm(rotation_targets,dim=2)
rotation_targets = rotation_targets/rotation_target_norms[:,:,None]
rotation_targets = rotation_targets.cuda(gpu)
loss_ = loss(rotation_predictions,rotation_targets)
print(loss_)