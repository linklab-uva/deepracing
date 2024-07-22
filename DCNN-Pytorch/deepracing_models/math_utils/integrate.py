import torch, torch.nn, torch.distributions
import numpy as np
import typing
from deepracing_models.math_utils.statistics import gaussian_pdf
class GaussianIntegral2D(torch.nn.Module):
    def __init__(self, gauss_order : int, intervalx = (-1, 1), intervaly = (-1, 1), batchdims : list[int] = [], requires_grad=False)-> None:
        super(GaussianIntegral2D, self).__init__()
        xhalfwidth = 0.5*(intervalx[1] - intervalx[0])
        xmean = 0.5*(intervalx[1] + intervalx[0])
        yhalfwidth = 0.5*(intervaly[1] - intervaly[0])
        ymean = 0.5*(intervaly[1] + intervaly[0])
        eta, weights = (torch.as_tensor(v) for v in np.polynomial.legendre.leggauss(gauss_order))
        gauss_pts_01 = torch.stack(torch.meshgrid(xhalfwidth*eta + xmean,yhalfwidth*eta + ymean,indexing='ij'), dim=0).reshape(2,-1)
        gauss_weights = (xhalfwidth*yhalfwidth)*(weights*weights[:,None]).ravel()
        self.eta_01 : torch.nn.Parameter = torch.nn.Parameter(gauss_pts_01, requires_grad=requires_grad)
        self.weights : torch.nn.Parameter = torch.nn.Parameter(gauss_weights, requires_grad=requires_grad)
    def forward(self, mvn : torch.distributions.MultivariateNormal, rotations : torch.Tensor, translations : torch.Tensor):
        gauss_pts = (rotations@self.eta_01).transpose(-2,-1) + translations[...,None,:]
        nbatchdims = translations.ndim - 1
        ls = np.linspace(0, nbatchdims-1, nbatchdims, dtype=np.int64)
        permute_idx = np.concatenate([[-2,], ls, [-1,]]).astype(ls.dtype)
        gaussian_pdf_vals = torch.exp(mvn.log_prob(gauss_pts.permute(*permute_idx)))
        permute_inv_idx =  np.concatenate([[-1], ls]).astype(ls.dtype)
        weights_exp = self.weights.tile(*np.ones_like(permute_inv_idx)).permute(*permute_inv_idx)
        return gauss_pts, gaussian_pdf_vals, torch.sum(gaussian_pdf_vals*weights_exp, dim=0).clip(0.0, 1.0)
    def __str__(self):
        return "Weights: %s.\n Eta: \n%s" % (str(self.weights.detach()), str(self.eta_01.transpose(-2,-1).detach()))

class GaussLegendre1D(torch.nn.Module):
    def __init__(self, gauss_order : int, interval = (-1, 1), requires_grad=False)-> None:
        super(GaussLegendre1D, self).__init__()
        intervalmean = 0.5*(interval[0] + interval[1])
        intervalhalfwidth = 0.5*(interval[1] - interval[0])
        eta, weights = (torch.as_tensor(v) for v in np.polynomial.legendre.leggauss(gauss_order))
        self.eta : torch.nn.Parameter = torch.nn.Parameter(intervalhalfwidth*eta + intervalmean, requires_grad=requires_grad)
        self.weights : torch.nn.Parameter = torch.nn.Parameter(intervalhalfwidth*weights, requires_grad=requires_grad)
    def forward(self, x : torch.Tensor):
        nbatchdims = x.ndim-1
        weights = self.weights.tile(*torch.ones(nbatchdims + 1, dtype=torch.int64))
        return torch.sum(weights*x, dim=-1)
        
def cumtrapz(y,x,initial=None):
    dx = x[:,1:]-x[:,:-1]
    avgy = 0.5*(y[:,1:]+y[:,:-1])
    #print("dx shape: ", dx.shape)
    #print("avgy shape: ", avgy.shape)
    mul = avgy*dx
   # print("mul shape: ", mul.shape)
    res = torch.cumsum(mul,1)
    #res = torch.stack([torch.cumsum(mul[:,:,i],dim=1) for i in range(y.shape[2])],dim=2)
    #print("res shape: ", res.shape)
    if initial is None:
        return res
    return torch.cat([initial,res],dim=1)
#come back to this later
def simpson(f_x, delta_x):
    numpoints = f_x.shape[1]
    if numpoints%2==0:
        raise ValueError("Number of points in f_x must be odd (for an even number of intervals as required by simpsons method)")
    if delta_x.shape[0]!=f_x.shape[0]:
        raise ValueError("Batch size of %d for delta_x but batch size of %d for f_x" %(delta_x.shape[0], f_x.shape[0]))
    simpsonintervals = numpoints -1

    simpsonscale = torch.ones(f_x.shape[0], numpoints, dtype=f_x.dtype, device=f_x.device)
    simpsonscale[:,list(range(1,simpsonintervals,2))] = 4.0
    simpsonscale[:,list(range(2,simpsonintervals,2))] = 2.0
    
    return (delta_x/3.0)*torch.sum(simpsonscale*f_x, dim=1)