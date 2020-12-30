import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LeakyReLU, ReLU, ELU
import numpy as np

def signedDistances(waypoints, boundarypoints, boundarynormals):
    batch_dimension = waypoints.shape[0]
    assert(boundarypoints.shape[0] == batch_dimension)
    assert(boundarynormals.shape[0] == batch_dimension)
    num_waypoints = waypoints.shape[1]
    num_boundary_points = boundarypoints.shape[1]
    point_dimension = waypoints.shape[2]
    distance_matrix = torch.cdist(waypoints, boundarypoints)
 #   print(distance_matrix.shape)
    closest_point_idx = torch.argmin(distance_matrix,dim=2)
    closest_boundary_points = torch.stack( [boundarypoints[i,closest_point_idx[i]] for i in range(batch_dimension)], dim=0)
    closest_boundary_normals = torch.stack( [boundarynormals[i,closest_point_idx[i]] for i in range(batch_dimension)], dim=0)
    delta_vecs = waypoints - closest_boundary_points
    return closest_point_idx, torch.sum(delta_vecs*closest_boundary_normals, dim=2)
class QuaternionDistance(nn.Module):
    def __init__(self):
        super(QuaternionDistance, self).__init__()
    #compute quaternion distance Along a tensor of shape [N, T, 4]
    #I am aware true quaternion distance has a factor of 2.0 out front.
    #But this is inteded to be used for neural network optimization
    #Where that extra linear factor is useless.
    def forward(self, inp, target):
        prod = torch.mul(inp,target)
        dot = torch.sum(prod, dim = 2) 
        dotabs = torch.abs(dot)
        dotabsthresh = torch.clamp(dotabs, 0.0, 1.0)
        acos = torch.acos(dotabsthresh)
        batched_sum = torch.sum(acos, dim = 1)
        return torch.sum( batched_sum )
class ScaledELU(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        super(ScaledELU, self).__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, inp):
        return torch.where(inp>=0.0, self.alpha*inp, torch.exp(self.beta*inp)-1.0)
class ScaledLeakyRelu(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        super(ScaledLeakyRelu, self).__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, inp):
        return torch.where(inp>=0.0, self.alpha*inp, self.beta*inp)
class ExpRelu(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        super(ExpRelu, self).__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, inp):
        return torch.where(inp>=0.0, self.alpha*torch.exp(inp) - 1.0, torch.exp(self.beta*inp) - 1.0)

class BoundaryLoss(nn.Module):
    def __init__( self, time_reduction="mean", batch_reduction="mean", alpha = 1.0, beta = 0.5, p = None, relu_type="Exp" ):
        super(BoundaryLoss, self).__init__()
        self.time_reduction = time_reduction
        self.batch_reduction = batch_reduction
        self.p = p
        if relu_type == "Elu":
            self.relu = ScaledELU(alpha=alpha, beta=beta)
        elif relu_type == "Leaky":
            self.relu = ScaledLeakyRelu(alpha = alpha, beta = beta)
        elif relu_type == "Exp":
            self.relu = ExpRelu(alpha = alpha, beta = beta)
        else:
            self.relu = ReLU()
    
    def forward(self, waypointslocal, boundarypoints, boundarynormals, posesglobal=None):
        batch_size = waypointslocal.shape[0]
        N = boundarypoints.shape[0]
        d = boundarypoints.shape[1]
        if posesglobal is not None:
            posesinv = torch.inverse(posesglobal)
            boundarypointslocal_ = torch.matmul(posesinv, boundarypoints)[:,0:3].transpose(1,2)
            boundarynormalslocal_ = torch.matmul(posesinv[:,0:3,0:3], self.boundarynormals).transpose(1,2)
        else:
            boundarypointslocal_ = boundarypoints
            boundarynormalslocal_ = boundarynormals
        # print("boundarynormalslocal_.shape: " + str(boundarynormalslocal_.shape))
        # print("boundarypointslocal_.shape: " + str(boundarypointslocal_.shape))
        # print("waypointslocal.shape: " + str(waypointslocal.shape))
        closest_point_idx, dot_prods = signedDistances(waypointslocal, boundarypointslocal_, boundarynormalslocal_)
        if self.p is None:
            dot_prods_relu = self.relu(dot_prods)
        elif self.p==2:
            dot_prods_relu = self.relu(torch.sign(dot_prods) * torch.square(dot_prods))
        else:
            dot_prods_relu = self.relu(torch.sign(dot_prods) * torch.pow(torch.abs(dot_prods),self.p))


        if self.time_reduction=="mean":
            dot_prod_redux = torch.mean(dot_prods_relu,dim=1)
        elif self.time_reduction=="sum":
            dot_prod_redux = torch.sum(dot_prods_relu,dim=1)
        elif self.time_reduction=="max":
            dot_prod_redux, max_idx = torch.max(dot_prods_relu,dim=1)
        elif self.time_reduction=="all":
            dot_prod_redux = dot_prods_relu
        else:
            raise ValueError("Unsupported time-wise reduction: %s" % (self.time_reduction,) )
      #  print("dot_prod_redux.shape: " + str(dot_prod_redux.shape))

        if self.batch_reduction=="mean":
            return closest_point_idx, torch.mean(dot_prod_redux)
        elif self.batch_reduction=="sum":
            return closest_point_idx, torch.sum(dot_prod_redux)
        elif self.batch_reduction=="max":
            return closest_point_idx, torch.max(dot_prod_redux)
        elif self.batch_reduction=="all":
            return closest_point_idx, dot_prod_redux
        else:
            raise ValueError("Unsupported batch-wise reduction: %s" % (self.batch_reduction,) )





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
        if self.p==2:
            squarednorms  = torch.sum(torch.square(diff),dim=self.dim)
        elif self.p%2 == 0:
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

class OtherAgentDistanceLoss(nn.Module):
    def __init__(self, alpha = 5.0,  beta : float = 1.0/np.pi):
        super(OtherAgentDistanceLoss, self).__init__()
        self.beta=beta
        self.alpha=alpha


    def forward(self, predicted_path, other_agent_paths, valid_index):
        if not (other_agent_paths.shape[2]==predicted_path.shape[1]):
            raise ValueError("Paths of other agents have %d points, but predicted path has %d points" % (other_agent_paths.shape[2],predicted_path.shape[1]))
        if not (other_agent_paths.shape[3]==predicted_path.shape[2]):
            raise ValueError("Paths of other agents are of dimension %d, but predicted path is of dimension %d" % (other_agent_paths.shape[3],predicted_path.shape[2]))
        batch_size = other_agent_paths.shape[0]
        num_points = other_agent_paths.shape[2]
        # diffs = other_agent_paths - predicted_path[:,None]
        # diffsquares = torch.square(diffs)
        # sumdiffsquares = torch.sqrt(torch.sum(diffsquares, dim=3))
        # exp = torch.exp(-self.beta*validmeandiffsquares)
        # meandiffsquare = torch.mean(sumdiffsquares, dim=2)
        distances = torch.norm(other_agent_paths - predicted_path[:,None], p=2, dim=3)
        meandistances = torch.mean(distances, dim=2)
        expdists = self.alpha*torch.exp(-self.beta*meandistances)
        validexpdists = expdists[valid_index]
       # validmeandistances = meandistances[meandistances==meandistances]
        # expdists = self.alpha*torch.exp(-self.beta*validmeandistances)
        rtn = torch.mean(validexpdists)
       # rtn = torch.mean(exp)
        return rtn

        
