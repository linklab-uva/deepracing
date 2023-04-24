from typing import List, Tuple, Union
from .bezier import bezierLsqfit
from .bezier import bezierM
from .bezier import bezierDerivative
from .fitting import pinv, fitAffine
from .bezier import bezierArcLength as bezierArcLength, BezierCurveModule, polynomialFormConversion

from .statistics import cov
from .integrate import cumtrapz, simpson

from .geometry import localRacelines

from . import bezier

import torch

import torch.nn

class CompositeBezierCurve(torch.nn.Module):
    def __init__(self, x : torch.Tensor, control_points : torch.Tensor) -> None:
        super(CompositeBezierCurve, self).__init__()

        self.control_points : torch.nn.Parameter =  torch.nn.Parameter(control_points.clone(), requires_grad=False)


        dx = x[1:] - x[:-1]
        if not torch.all(dx>0):
            raise ValueError("x values must be in ascending order")
        
        self.dx : torch.nn.Parameter =  torch.nn.Parameter(dx.clone(), requires_grad=False)
        self.xstart_vec : torch.nn.Parameter =  torch.nn.Parameter(x[:-1].clone(), requires_grad=False)
        self.xend_vec : torch.nn.Parameter =  torch.nn.Parameter((x[:-1]+dx).clone(), requires_grad=False)

        self.x : torch.Tensor = torch.nn.Parameter(x.clone(), requires_grad=False)


    def forward(self, x_eval : torch.Tensor):
        x_true = x_eval%self.xend_vec[-1]
        curve_indices = torch.searchsorted(self.xstart_vec, x_true, right=True)-1
        idx : torch.Tensor = torch.linspace(0, curve_indices.shape[0]-1, steps=curve_indices.shape[0], dtype=torch.int64)
        transition_booleans = curve_indices[idx]!=curve_indices[(idx+1)%idx.shape[0]]
        transition_indices = torch.where(transition_booleans)[0]+1

        xindex = transition_indices[0]
        curveindex = curve_indices[0]
        x_samp = x_true[0:xindex]
        curve = self.control_points[curveindex]
        xmin = self.xstart_vec[curveindex]
        s_samp = (x_samp - xmin)/self.dx[curveindex]
        Msamp = bezierM(s_samp.unsqueeze(0), self.control_points.shape[1]-1)[0]
        blocks : List[torch.Tensor] = [torch.matmul(Msamp, curve)]
        for i in range(transition_indices.shape[0]-1):
            x_samp = x_true[transition_indices[i]:transition_indices[i+1]]
            curveindex = curve_indices[transition_indices[i]]
            curve = self.control_points[curveindex]
            xmin = self.xstart_vec[curveindex]
            s_samp = (x_samp - xmin)/self.dx[curveindex]
            Msamp = bezierM(s_samp.unsqueeze(0), self.control_points.shape[1]-1)[0]
            blocks.append(torch.matmul(Msamp, curve))
        xindex = transition_indices[-1]
        curveindex = curve_indices[-1]
        x_samp = x_true[xindex:]
        curve = self.control_points[curveindex]
        xmin = self.xstart_vec[curveindex]
        s_samp = (x_samp - xmin)/self.dx[curveindex]
        Msamp = bezierM(s_samp.unsqueeze(0), self.control_points.shape[1]-1)[0]
        blocks.append(torch.matmul(Msamp, curve))
        return torch.cat(blocks, dim=0)
    def derivative(self):
        control_points_detached = self.control_points.detach()
        control_point_deltas : torch.Tensor = (control_points_detached.shape[1]-1)*(control_points_detached[:,1:] - control_points_detached[:,:-1])/self.dx.detach()[:,None,None]
        return CompositeBezierCurve(self.x.detach(), control_point_deltas)

def compositeBezierSpline(x : torch.Tensor, Y : torch.Tensor, boundary_conditions : Union[str,torch.Tensor] = "periodic"):
    if boundary_conditions=="periodic":
        return bezier.compositeBezierSpline_periodic_(x,Y)
    else:
        return bezier.compositeBezierSpline_with_boundary_conditions_(x, Y, boundary_conditions)


def closedPathAsBezierSpline(Y : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    euclidean_distances : torch.Tensor = torch.zeros_like(Y[:,0])
    delta_euclidean_distances = torch.norm( Y[1:]-Y[:-1] , dim=1 )
    euclidean_distances[1:] = torch.cumsum( delta_euclidean_distances, 0 )
    euclidean_spline = bezier.compositeBezierSpline_periodic_(euclidean_distances,Y)
    arclengths = torch.zeros_like(Y[:,0])
    _, _, _, _, distances = bezierArcLength(euclidean_spline, N=2, simpsonintervals=20)
    arclengths[1:] = torch.cumsum(distances[:,-1], 0)
    return arclengths, bezier.compositeBezierSpline_periodic_(arclengths,Y)