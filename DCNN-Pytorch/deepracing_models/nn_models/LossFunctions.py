import torch
import torch.nn as nn
import torch.nn.functional as F
class QuaternionDistance(nn.Module):
    def __init__(self):
        super(QuaternionDistance, self).__init__()
    #compute quaternion distance Along a tensor of shape [N, T, 4]
    #I am aware true quaternion distance has a factor of 2.0 out front.
    #But this is inteded to be used for neural network optimization
    #Where that extra linear factor is useless.
    def forward(self, input, target):
        prod = torch.mul(input,target)
        dot = torch.sum(prod,dim=2) 
       # print(dot.shape)
        dotabs = torch.abs(dot)
        dotabsthresh = torch.clamp(dotabs, 0.0, 1.0)
        acos = torch.acos(dotabsthresh)
       # print(acos.shape)
        batched_sum = torch.sum(acos, dim = 1)
        #print(batched_sum.shape)
        return torch.sum( batched_sum )