import numpy as np
import torch
from scipy.special import comb as nChoosek
Mtk = lambda k, n, t: (t**(k))*((1-t)**(n-k))*nChoosek(n,k)
def pinv(A):
    """
    Return the batchwise pseudoinverse of A

    PyTorch's SVD function now supports batchwise solving, so this is pretty easy.
    """
    batch, rows, cols = A.size()
    U,S,V = torch.svd(A)
    sinv = 1/S
    sinv[sinv == float("Inf")] = 0
    return torch.matmul(torch.matmul(V,torch.diag_embed(sinv).transpose(1,2)),U.transpose(1,2))
def bezierM(t,n):
    # M = torch.zeros(t.shape[0],t.shape[1],n+1).type_as(t)
    # for j in range(M.shape[0]):
    #     M[j]=torch.stack([Mtk(i,n,t[j]) for i in range(n+1)],dim=1)
    # return M
    return torch.stack([Mtk(k,n,t) for k in range(n+1)],dim=2)
def evalBezier(M,control_points):
    return torch.matmul(M,control_points)
def bezierDerivative(control_points,n,t, order=1):
    Mderiv = bezierM(t,n-order)
    pdiff =  control_points[:,1:] - control_points[:,:-1]
    #print(Mderiv.shape)
    #print(pdiff.shape)
    return n*torch.matmul(Mderiv, pdiff[:,:(n-order+1)])
def bezierLsqfit(points,t, n):
    M = bezierM(t,n)
    return M, torch.matmul(pinv(M), points)