import torch
import numpy as np
from scipy.special import comb as nChoosek

def localRacelines(poses_global : torch.Tensor, raceline_global : torch.Tensor, lookahead_distance : float):
    if not poses_global.shape[1:] == torch.Size([4,4]):
        raise ValueError("poses_global must be a batch of [4x4] tensors")
    