import torch, torch.nn# , torch.distributions
import numpy as np
# import typing
# from deepracing_models.math_utils.statistics import gaussian_pdf
class GaussianIntegralCirc(torch.nn.Module):
    def __init__(self, gauss_order : int, radius : float = 1.0, requires_grad=False)-> None:
        super(GaussianIntegralCirc, self).__init__()
        Rhalfwidth = Rmean = 0.5*radius
        thetahalfwidth = thetamean = float(np.pi)
        eta, weights = (torch.as_tensor(v) for v in np.polynomial.legendre.leggauss(gauss_order))
        self.weights = torch.nn.Parameter((weights*weights[:,None]).ravel(), requires_grad=requires_grad)
        gauss_pts_rtheta = torch.stack(torch.meshgrid(Rhalfwidth*eta + Rmean,thetahalfwidth*eta + thetamean,indexing='ij'), dim=0).reshape(2,-1)
        # print("gauss_pts_rtheta:", gauss_pts_rtheta)
        self.gauss_pts_rtheta : torch.nn.Parameter = torch.nn.Parameter(gauss_pts_rtheta.transpose(0,1), requires_grad=requires_grad)
        gauss_pts_xy = torch.stack([gauss_pts_rtheta[0]*torch.cos(gauss_pts_rtheta[1]), gauss_pts_rtheta[0]*torch.sin(gauss_pts_rtheta[1])], dim=1)
        self.gauss_pts_xy : torch.nn.Parameter = torch.nn.Parameter(gauss_pts_xy, requires_grad=requires_grad)
        self.outer_factor = torch.nn.Parameter(torch.as_tensor(Rhalfwidth*thetahalfwidth), requires_grad=requires_grad)

    def forward(self, target_means : torch.Tensor, target_stdev_inverse_matrices : torch.Tensor, target_logstdevs : torch.Tensor):
        diff = self.gauss_pts_xy - target_means[...,None,:]
        points01 = (target_stdev_inverse_matrices@diff.transpose(-2,-1)).transpose(-2,-1)
        points01square = points01.square()
        log_pdf_vals2 = -0.5*(points01square.sum(dim=-1)) - target_logstdevs[:,None]
        pdfvals = log_pdf_vals2.exp()
        cdfvals = (self.outer_factor*((pdfvals*self.gauss_pts_rtheta[None,:,0])*self.weights[None]).sum(dim=1)).clip(0.0, 1.0)
        return pdfvals, cdfvals

                
class GaussianIntegral2D(torch.nn.Module):
    def __init__(self, gauss_order : int, intervalx = (-1, 1), intervaly = (-1, 1), requires_grad=False)-> None:
        super(GaussianIntegral2D, self).__init__()
        xhalfwidth = 0.5*(intervalx[1] - intervalx[0])
        xmean = 0.5*(intervalx[1] + intervalx[0])
        yhalfwidth = 0.5*(intervaly[1] - intervaly[0])
        ymean = 0.5*(intervaly[1] + intervaly[0])
        eta, weights = (torch.as_tensor(v) for v in np.polynomial.legendre.leggauss(gauss_order))
        gauss_pts_01 = torch.stack(torch.meshgrid(xhalfwidth*eta + xmean,yhalfwidth*eta + ymean,indexing='ij'), dim=0).reshape(2,-1)
        gauss_weights = (weights*weights[:,None]).ravel()
        self.outer_factor = torch.nn.Parameter(torch.as_tensor(xhalfwidth*yhalfwidth), requires_grad=False) #(xhalfwidth*yhalfwidth)*
        self.eta_01 : torch.nn.Parameter = torch.nn.Parameter(gauss_pts_01.transpose(0,1), requires_grad=requires_grad)
        self.weights : torch.nn.Parameter = torch.nn.Parameter(gauss_weights, requires_grad=requires_grad)
    def forward(self, target_means : torch.Tensor, target_stdev_inverse_matrices : torch.Tensor, target_logstdevs : torch.Tensor,
                rotations : torch.Tensor, translations : torch.Tensor):
        prebatched = target_means.ndim > 3
        eta_01_exp = self.eta_01[None,:,None].unsqueeze(-1)
        gauss_pts = (rotations[:,None]@eta_01_exp).squeeze(-1) + translations[:,None]
        # print("gauss_pts.shape:", gauss_pts.shape)
        # print("target_means.shape:", target_means.shape)
        target_means_exp = target_means.unsqueeze(1) if prebatched else target_means
        diff = gauss_pts.unsqueeze(-2) - target_means_exp
        # print("diff.shape:", diff.shape)
        target_stdev_inverse_matrices_exp = target_stdev_inverse_matrices.unsqueeze(1) if prebatched else target_stdev_inverse_matrices
        # print("diff.shape:", diff.shape)
        # print("target_stdev_inverse_matrices_exp.shape:", target_stdev_inverse_matrices_exp.shape)
        # a = torch.sum(target_stdev_inverse_matrices_exp[...,0,:]*diff, dim=-1)
        # b = torch.sum(target_stdev_inverse_matrices_exp[...,1,:]*diff, dim=-1)
        # squaresums = a.square() + b.square()
        points01 = (target_stdev_inverse_matrices_exp@diff.unsqueeze(-1)).squeeze(-1)
        points01square = points01.square()
        squaresums = points01square.sum(dim=-1)

        # print("points01square.shape:", points01square.shape)
        # print("target_logstdevs.shape:", target_logstdevs.shape)
        target_logstdevs_exp = target_logstdevs.unsqueeze(1) if prebatched else target_logstdevs
        log_pdf_vals2 = -0.5*squaresums - target_logstdevs_exp#[None,:,None]
        pdfvals = log_pdf_vals2.exp()
        cdfvals = (self.outer_factor*(pdfvals*self.weights[None,:,None,None]).sum(dim=1)).clip(0.0, 1.0)
        return gauss_pts, pdfvals, cdfvals
    def __str__(self):
        return "Weights: %s.\n Eta: \n%s" % (str(self.weights.detach()), str(self.eta_01.transpose(-2,-1).detach()))

class GaussLegendre1D(torch.nn.Module):
    def __init__(self, gauss_order : int, interval = (-1, 1), requires_grad=False)-> None:
        super(GaussLegendre1D, self).__init__()
        intervalmean = 0.5*(interval[0] + interval[1])
        intervalhalfwidth=0.5*(interval[1] - interval[0])
        self.intervalhalfwidth = torch.nn.Parameter(torch.as_tensor(intervalhalfwidth), requires_grad=False)
        eta, weights = (torch.as_tensor(v) for v in np.polynomial.legendre.leggauss(gauss_order))
        self.eta : torch.nn.Parameter = torch.nn.Parameter(intervalhalfwidth*eta + intervalmean, requires_grad=requires_grad)
        self.weights : torch.nn.Parameter = torch.nn.Parameter(weights, requires_grad=requires_grad) #intervalhalfwidth*
    def forward(self, x : torch.Tensor):
        # nbatchdims = x.ndim-1
        # weights = self.weights.tile(*torch.ones(nbatchdims + 1, dtype=torch.int64))
        # weights = self.weights #.tile(*[1 for _ in range(nbatchdims+1)])
        # return self.intervalhalfwidth*torch.sum(weights*x, dim=-1)
        return self.intervalhalfwidth*torch.sum(self.weights*x, dim=-1)
        
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