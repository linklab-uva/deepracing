import torch
import numpy as np
from scipy.special import comb as nChoosek
import deepracing_models.math_utils.bezier as bezier_utils
def localRacelines(poses_global : torch.Tensor, raceline_global : torch.Tensor, lookahead_distance : float):
    if not poses_global.shape[1:] == torch.Size([4,4]):
        raise ValueError("poses_global must be a batch of [4x4] tensors")
    if not raceline_global.shape[0] == 4:
        raise ValueError("raceline_global must be a raceline matrix of size 4xN")
    poses_global_inv = torch.inverse(poses_global)
    raceline_locals = torch.matmul(poses_global_inv, raceline_global)
    meandiff = torch.mean( torch.linalg.norm(raceline_global[0:3,1:] - raceline_global[0:3,:-1], ord=2, dim=1) )
    lookahead_indices = int(round(lookahead_distance/meandiff.item()))
    return raceline_locals
    