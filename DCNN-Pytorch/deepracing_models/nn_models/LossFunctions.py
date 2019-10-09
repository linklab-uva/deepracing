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
        dot = torch.sum(prod, dim = 2) 
        dotabs = torch.abs(dot)
        dotabsthresh = torch.clamp(dotabs, 0.0, 1.0)
        acos = torch.acos(dotabsthresh)
        batched_sum = torch.sum(acos, dim = 1)
        return torch.sum( batched_sum )
class SquaredLpNormLoss(nn.Module):
    '''
    Computes the squared Lp norm between predictions and target along a time axis .
    Args:
      p: Which norm to compute (default is 2-norm)
      dim: Dimension along which to compute the squared norm
      time_reduction: How to reduce along the time axis (assumed to be dimension dim-1 in the original tensors) ('mean' or 'sum')
      batch_reduction: How to reduce along the batch axis (assumed to be dimension 0) ('mean' or 'sum')
    '''
    def __init__(self, time_reduction="mean", batch_reduction="mean", p = 2, dim = 2, timewise_weights = None):
        super(SquaredLpNormLoss, self).__init__()
        self.batch_reduction=batch_reduction
        self.time_reduction=time_reduction
        self.p=p
        self.dim=dim
        if not (timewise_weights is None):
            self.timewise_weights=nn.Parameter(timewise_weights, requires_grad=False)
        else:
            self.timewise_weights=None
        #self.pairwise_dist = nn.PairwiseDistance(p=self.p)
    def forward(self, predictions, targets):
        diff = predictions - targets
        if self.p%2 == 0:
            squarednorms  = torch.sum(torch.pow(diff,self.p),dim=self.dim)
        else:
            squarednorms  = torch.sum(torch.abs(torch.pow(diff,self.p)),dim=self.dim)


        if not (self.timewise_weights is None):
            squarednorms = self.timewise_weights[:,None]*squarednorms
        if self.time_reduction=="mean":
            means = torch.mean(squarednorms,dim=self.dim-1)
        elif self.time_reduction=="sum":
            means = torch.sum(squarednorms,dim=self.dim-1)
        else:
            means = squarednorms
        if self.batch_reduction=="mean":
            return torch.mean(means)
        elif self.batch_reduction=="sum":
            return torch.sum(means)
        else:
            return means
class TaylorSeriesLinear(nn.Module):
    def __init__(self, reduction="mean"):
        super(TaylorSeriesLinear, self).__init__()
        self.reduction=reduction
    def forward(self, position, velocity, time, acceleration=None):
        position_diff = position[:,1:,:]-position[:,:-1,:]
        dt=time[:,1:]-time[:,:-1]
        vel_est = position_diff/dt[:,:,None]
        squarednorms = torch.norm((velocity[:,:-1,:] - vel_est),dim=2, p=2)
        if self.reduction=="mean":
            return torch.mean(squarednorms)
        else:
            return torch.sum(squarednorms)
