import torch
import torch.nn as nn
import torch.nn.functional as F
class QuaternionDistance(nn.Module):
    def __init__(self):
        super(QuaternionDistance, self).__init__()
    def forward(self, input, target):
        prod = torch.mul(input,target)
        dot = torch.sum(prod,dim=2) 
       # print(dot.shape)
        acos = torch.acos(dot)
       # print(acos.shape)
        batched_sum = torch.sum(acos, dim = 1)
        #print(batched_sum.shape)
        return torch.sum( batched_sum )