from typing import List, Tuple, Union
from .bezier import bezierLsqfit
from .bezier import bezierM
from .bezier import bezierDerivative
from .bezier import bezierPolyRoots
from .fitting import pinv, fitAffine
from .bezier import bezierArcLength as bezierArcLength, BezierCurveModule, polynomialFormConversion, elevateBezierOrder

from .statistics import cov
from .integrate import cumtrapz, simpson

from .geometry import localRacelines

from . import bezier

import torch

import torch.nn

def compositeBezierSpline(x : torch.Tensor, Y : torch.Tensor, boundary_conditions : Union[str,torch.Tensor] = "periodic"):
    if boundary_conditions=="periodic":
        return bezier.compositeBezierSpline_periodic_(x,Y)
    elif type(boundary_conditions)==torch.Tensor:
        return bezier.compositeBezierSpline_with_boundary_conditions_(x, Y, boundary_conditions)
    else:
        raise ValueError("Currently, only supported values for boundary_conditions are the string \"periodic\" or a torch.Tensor of boundary values")


def closedPathAsBezierSpline(Y : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if not torch.all(Y[0]==Y[-1]):
        Yaug : torch.Tensor = torch.cat([Y, Y[0].unsqueeze(0)], dim=0)
    else:
        Yaug : torch.Tensor = Y
    euclidean_distances : torch.Tensor = torch.zeros_like(Yaug[:,0])
    delta_euclidean_distances = torch.norm( Yaug[1:]-Yaug[:-1] , dim=1 )
    euclidean_distances[1:] = torch.cumsum( delta_euclidean_distances, 0 )
    euclidean_spline = compositeBezierSpline(euclidean_distances,Yaug,boundary_conditions="periodic")
    arclengths = torch.zeros_like(Yaug[:,0])
    _, _, _, _, distances = bezierArcLength(euclidean_spline, N=2, simpsonintervals=20)
    arclengths[1:] = torch.cumsum(distances[:,-1], 0)
    return arclengths, compositeBezierSpline(arclengths,Yaug,boundary_conditions="periodic")


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
        idx_forward : torch.Tensor = torch.linspace(1, curve_indices.shape[0], steps=curve_indices.shape[0], dtype=torch.int64)
        idx_forward[-1]=0
        transition_booleans = curve_indices!=curve_indices[idx_forward]
        transition_indices = torch.where(transition_booleans)[0]+1

        x_final_idx = transition_indices[0] if transition_indices.shape[0]>0 else None
        curveindex = curve_indices[0]
        x_samp = x_true[0:x_final_idx]
        curve = self.control_points[curveindex]
        xmin = self.xstart_vec[curveindex]
        s_samp = (x_samp - xmin)/self.dx[curveindex]
        Msamp = bezierM(s_samp.unsqueeze(0), self.control_points.shape[1]-1)[0]
        blocks : List[torch.Tensor] = [torch.matmul(Msamp, curve)]
        for i in range(transition_indices.shape[0]):
            x_final_idx, curveindex = (transition_indices[i+1], curve_indices[transition_indices[i]]) if i<transition_indices.shape[0]-1 else (None,curve_indices[-1])
            x_samp = x_true[transition_indices[i]:x_final_idx]
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

class SimplePathHelper(torch.nn.Module):
    def __init__(self, points : torch.Tensor, dr_samp : float) -> None:
        super(SimplePathHelper, self).__init__()
        arclengths_, curve_control_points_ = closedPathAsBezierSpline(points)
        self.__arclengths_in__ : torch.nn.Parameter = torch.nn.Parameter(arclengths_.clone(), requires_grad=False)
        self.__points_in__ : torch.nn.Parameter = torch.nn.Parameter(points.clone(), requires_grad=False)

        self.__curve__ : CompositeBezierCurve = CompositeBezierCurve(arclengths_, curve_control_points_).requires_grad_(False)
        self.__curve_deriv__ : CompositeBezierCurve = self.__curve__.derivative().requires_grad_(False)
        self.__curve_2nd_deriv__ : CompositeBezierCurve = self.__curve_deriv__.derivative().requires_grad_(False)

        self.__r_samp__ : torch.nn.Parameter = torch.nn.Parameter(torch.arange(0.0, arclengths_[-1], step=dr_samp, dtype=points.dtype, device=points.device), requires_grad=False)
        points_samp : torch.Tensor = self.__curve__(self.__r_samp__).detach().clone()
        self.__points_samp__ : torch.nn.Parameter = torch.nn.Parameter(points_samp, requires_grad=False)
        tangents_samp : torch.Tensor = self.__curve_deriv__(self.__r_samp__).detach().clone()
        self.__tangents_samp__ : torch.nn.Parameter = torch.nn.Parameter(tangents_samp, requires_grad=False)
        normals_samp = tangents_samp[:,[1,0]].clone()
        normals_samp[:,0]*=-1.0
        self.__normals_samp__ : torch.nn.Parameter = torch.nn.Parameter(normals_samp, requires_grad=False)





    def offset_points(self, left_offset : float, right_offset: float) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.__points_samp__ + self.__normals_samp__*left_offset, self.__points_samp__ - self.__normals_samp__*right_offset
    def eval(self, s : torch.Tensor):
        return self.__curve__(s)
    def tangent(self, s : torch.Tensor):
        return self.__curve_deriv__(s)
    def forward(self, s : torch.Tensor):
        return self.__curve__(s)#, self.__curve_deriv__(s), self.__curve_2nd_deriv__(s)
    

def closestPointToPathNaive(path : SimplePathHelper, p_query : torch.Tensor):
    iclosest = torch.argmin(torch.norm(path.__points_samp__ - p_query, p=2, dim=1))
    return path.__r_samp__[iclosest]%path.__curve__.xend_vec[-1], path.__points_samp__[iclosest]
def closestPointToPath(path : SimplePathHelper, p_query : torch.Tensor, s0 : Union[None, torch.Tensor] = None, lr = 1.0, max_iter = 10000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if s0 is None:
        iclosest = torch.argmin(torch.norm(path.__points_samp__ - p_query, p=2, dim=1))
        s_euclidean_approx = path.__r_samp__[iclosest]%path.__curve__.xend_vec[-1]
        s_optim : torch.nn.parameter.Parameter = torch.nn.parameter.Parameter(torch.as_tensor([s_euclidean_approx], dtype=path.__r_samp__.dtype, device=path.__r_samp__.device), requires_grad=True)
    else:
        s_optim : torch.nn.parameter.Parameter = torch.nn.parameter.Parameter(torch.as_tensor([s0%path.__curve__.xend_vec[-1]], dtype=path.__r_samp__.dtype, device=path.__r_samp__.device)%path.__curve__.xend_vec[-1], requires_grad=True)
    sgd = torch.optim.SGD([s_optim], lr)
    s_init = s_optim[0].detach().clone()
    x0 = path(s_optim.detach().clone())
    x0 : torch.Tensor = x0[0]
    lossprev : torch.Tensor = None
    loss : torch.Tensor = None
    for asdf in range(max_iter):
        x_curr = path(s_optim)
        delta = p_query - x_curr[0]
        loss = torch.norm(delta, p=2)
        sgd.zero_grad()
        loss.backward() 
        sgd.step()
        if lossprev is not None:
            delta_loss = loss - lossprev
            if torch.abs(delta_loss)<1E-8:
                break
        lossprev = loss
    return s_init, x0, s_optim.detach()[0], x_curr.detach()
