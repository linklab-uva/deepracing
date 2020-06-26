import numpy as np
import torch
from scipy.special import comb as nChoosek
def Mtk(k,n,t):
    return torch.pow(t,k)*torch.pow(1-t,(n-k))*nChoosek(n,k)
def pinv(A, minimum_singular_value = 0.0):
    """
    Return the batchwise pseudoinverse of A

    PyTorch's SVD function now supports batchwise solving, so this is pretty easy.
    """
   # batch, rows, cols = A.size()
    U,S,V = torch.svd(A)
    sinv =  torch.where(S > minimum_singular_value, 1/S, torch.zeros_like(S))
    #sinv = 1/S
    #sinv[sinv == float("Inf")] = 0
    return torch.matmul(torch.matmul(V,torch.diag_embed(sinv).transpose(1,2)),U.transpose(1,2))
def bezierM(t,n):
    # M = torch.zeros(t.shape[0],t.shape[1],n+1).type_as(t)
    # for j in range(M.shape[0]):
    #     M[j]=torch.stack([Mtk(i,n,t[j]) for i in range(n+1)],dim=1)
    # return M
    return torch.stack([Mtk(k,n,t) for k in range(n+1)],dim=2)
def evalBezier(M,control_points):
    return torch.matmul(M,control_points)
    
def bezierDerivative(control_points, t = None, M = None, order = 1 ):
    if (bool(t is not None) ^ bool(M is not None)):
        n = control_points.shape[1]-1
        if t is not None:
            Mderiv = bezierM(t,n-order)
        else:
            Mderiv = M
        pdiff =  control_points[:,1:] - control_points[:,:-1]
        for i in range(1,order):
            pdiff =  pdiff[:,1:] - pdiff[:,:-1]
        factor = torch.prod(torch.linspace(n,n-order+1,order))
        return Mderiv, factor*torch.matmul(Mderiv, pdiff)
    else:
        raise ValueError("One of t or M must be set, but not both")

def bezierLsqfit(points, n, t = None, M = None, built_in_lstq=False):
    if ((t is None) and (M is None)) or ((t is not None) and (M is not None)):
        raise ValueError("One of t or M must be set, but not both")
    if M is None:
        M_ = bezierM(t,n)
    else:
        M_ = M
    batch = M_.shape[0]
    if built_in_lstq:
        return M_, torch.stack([torch.lstsq(points[i], M_[i])[0][0:n+1] for i in range(batch)],dim=0)
    else:
        return M_, torch.matmul(pinv(M_), points)