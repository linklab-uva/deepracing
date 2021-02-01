import torch
import numpy as np

def pinv(A, minimum_singular_value = 0.0):
    """
    Return the (optionally batchwise) pseudoinverse of A

    PyTorch's SVD function now supports batchwise solving, so this is pretty easy.
    """
   # batch, rows, cols = A.size()
    U,S,V = torch.svd(A)
    sinv =  torch.where(S > minimum_singular_value, 1/S, torch.zeros_like(S))
    #sinv = 1/S
    #sinv[sinv == float("Inf")] = 0
    batch=A.ndim>2
    if batch:
        return torch.matmul(torch.matmul(V,torch.diag_embed(sinv).transpose(-2,-1)),U.transpose(-2,-1))
    else:
        return torch.matmul(torch.matmul(V,torch.diag_embed(sinv).t()),U.t())
def fitAffine(p0 : torch.Tensor, pf: torch.Tensor):
    batch=p0.ndim>2
    if batch:
        paug = torch.cat([p0, torch.ones_like(p0[:,:,0]).unsqueeze(2)], dim=2)
    else:
        paug = torch.cat([p0, torch.ones_like(p0[:,0]).unsqueeze(1)], dim=1)
    return paug, torch.matmul(pinv(paug), pf)
