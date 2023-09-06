from typing import List, Tuple, Union
import typing
from .bezier import bezierLsqfit
from .bezier import bezierM
from .bezier import bezierDerivative
from .bezier import bezierPolyRoots
from .bezier import evalBezier, evalBezierSinglePoint
from .bezier import bezierArcLength, compositeBezierSpline, compositeBezierAntiderivative, compositeBezierFit
from .bezier import closedPathAsBezierSpline, polynomialFormConversion, elevateBezierOrder, compositeBezierEval
from .fitting import pinv, fitAffine
from .bezier import BezierCurveModule

from .statistics import cov
from .integrate import cumtrapz, simpson

from .geometry import localRacelines

from . import bezier


import torch

import torch.nn

class CompositeBezierCurve(torch.nn.Module):
    def __init__(self, x : torch.Tensor, control_points : torch.Tensor, order : int = 0) -> None:
        super(CompositeBezierCurve, self).__init__()

        self.control_points : torch.nn.Parameter =  torch.nn.Parameter(control_points.clone(), requires_grad=False)


        dx = x[1:] - x[:-1]
        if not torch.all(dx>0):
            raise ValueError("x values must be in ascending order")
        
        self.dx : torch.nn.Parameter =  torch.nn.Parameter(dx.clone(), requires_grad=False)
        self.xstart_vec : torch.nn.Parameter =  torch.nn.Parameter(x[:-1].clone(), requires_grad=False)
        self.xend_vec : torch.nn.Parameter =  torch.nn.Parameter(x[1:].clone(), requires_grad=False)

        self.x : torch.nn.Parameter = torch.nn.Parameter(x.clone(), requires_grad=False)

        self.d : torch.nn.Parameter = torch.nn.Parameter(torch.as_tensor(self.control_points.shape[-1], dtype=torch.int64), requires_grad=False)

        self.order : torch.nn.Parameter = torch.nn.Parameter(torch.as_tensor(order, dtype=torch.int64), requires_grad=False)

        self.bezier_order : torch.nn.Parameter = torch.nn.Parameter(torch.as_tensor(self.control_points.shape[-2]-1, dtype=torch.int64), requires_grad=False)




    def forward(self, x_eval : torch.Tensor, idxbuckets : typing.Union[None,torch.Tensor] = None):
        x_true = (x_eval%self.xend_vec[-1]).view(1,-1)
        # if imin is None:
        #     imin_ = (torch.bucketize(x_true.detach(), self.xend_vec.detach(), right=False) ) #% self.xend_vec[-1]
        # else:
        #     imin_ = imin
        # xstart_select = self.xstart_vec[imin_]
        # dx_select = self.dx[imin_]
        # points_select = self.control_points[imin_]
        # s_select = (x_true - xstart_select)/dx_select
        # return evalBezierSinglePoint(s_select, points_select), imin_
        evalout, idxmin = compositeBezierEval(self.xstart_vec.unsqueeze(0), self.dx.unsqueeze(0), self.control_points.unsqueeze(0), x_true, idxbuckets=idxbuckets)
        evalrtn = evalout.view(list(x_eval.shape) + [self.d.item()])
        return evalrtn, idxmin.view(x_eval.shape)
    def derivative(self):
        control_points_detached = self.control_points.detach()
        control_point_deltas : torch.Tensor = self.bezier_order*(control_points_detached[:,1:] - control_points_detached[:,:-1])/self.dx.detach()[:,None,None]
        return CompositeBezierCurve(self.x.detach().clone(), control_point_deltas, order=self.order+1)

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
        
        tup = self.__curve__(self.__r_samp__)
        points_samp : torch.Tensor = tup[0].detach().clone()
        self.__points_samp__ : torch.nn.Parameter = torch.nn.Parameter(points_samp, requires_grad=False)

        tup = self.__curve_deriv__(self.__r_samp__)
        tangents_samp : torch.Tensor = tup[0].detach().clone()
        self.__tangents_samp__ : torch.nn.Parameter = torch.nn.Parameter(tangents_samp, requires_grad=False)

        normals_samp = tangents_samp[:,[1,0]].clone()
        normals_samp[:,0]*=-1.0
        self.__normals_samp__ : torch.nn.Parameter = torch.nn.Parameter(normals_samp, requires_grad=False)
    def offset_points(self, left_offset : float, right_offset: float) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.__points_samp__ + self.__normals_samp__*left_offset, self.__points_samp__ - self.__normals_samp__*right_offset
    def tangent(self, s : torch.Tensor):
        derivs, _ = self.__curve_deriv__(s)
        return derivs
    def forward(self, s : torch.Tensor, deriv=False):
        positions, idxbuckets = self.__curve__(s)
        if deriv:
            derivs, _ = self.__curve_deriv__(s, idxbuckets=idxbuckets)
            return positions, derivs
        return positions, None
    
    def x_axis_intersection(self, Pquery : torch.Tensor, Rquery : torch.Tensor):
        
        batchdim = Pquery.shape[0]
        control_points = self.__curve__.control_points
        arclengths_start = self.__curve__.x
        delta_arclengths = self.__curve__.dx
        
        rotmat = Rquery.transpose(-2,-1).to(control_points.device).to(control_points.dtype)
        pquery = Pquery.to(control_points.device).to(control_points.dtype)
        ptransform = -torch.matmul(rotmat, pquery.unsqueeze(-1)).squeeze(-1)

        control_points_exp = control_points.unsqueeze(0).expand([batchdim] + list(control_points.shape))
        control_points_exp_flat = control_points_exp.view(batchdim, -1, control_points.shape[-1])
        control_points_transformed_flat = torch.matmul(rotmat, control_points_exp_flat.transpose(-2,-1)).transpose(-2,-1) + ptransform[:,None]

        control_points_transformed = control_points_transformed_flat.view(batchdim, control_points.shape[0], control_points.shape[1], control_points.shape[2])


        kbezier = control_points.shape[1] - 1
        xbezier = control_points_transformed[:,:,:,0]
        polynom_roots = bezierPolyRoots(xbezier.reshape(-1, kbezier+1)).view(batchdim, control_points.shape[0], kbezier)

        matchmask = (torch.abs(polynom_roots.imag)<1E-5)*(polynom_roots.real>0.0)*(polynom_roots.real<1.0)

        idx=torch.arange(0, control_points.shape[0], step=1, dtype=torch.int64, device=control_points.device)
        
        selection_all = (torch.sum(matchmask, dim=-1)>=1)
        
        rintersect = torch.zeros(batchdim, device=control_points.device, dtype=control_points.dtype)
        for i in range(batchdim):
            selection = selection_all[i]
            candidates_idx = idx[selection]
            candidates = control_points_transformed[i,candidates_idx]
            candidates_polyroots = polynom_roots[i,candidates_idx]
            candidates_rstart = arclengths_start[candidates_idx]
            candidates_dr = delta_arclengths[candidates_idx]
            norms = torch.norm(candidates[:,[0,-1]], p=2.0, dim=2)
            norm_means = torch.mean(norms, dim=1)
            imin = torch.argmin(norm_means)

            correctroots = candidates_polyroots[imin]
            correctsval = correctroots[(torch.abs(correctroots.imag)<1E-5)*(correctroots.real>0.0)*(correctroots.real<1.0)].real.item()
            correctdr = candidates_dr[imin]

            correctrstart = candidates_rstart[imin]

            rintersect[i] = correctrstart + correctsval*correctdr
        return rintersect
        # print(rintersect)


        

    

def closestPointToPathNaive(path : SimplePathHelper, p_query : torch.Tensor):
    iclosest = torch.argmin(torch.norm(path.__points_samp__ - p_query, p=2, dim=1))
    return path.__r_samp__[iclosest]%path.__curve__.xend_vec[-1], path.__points_samp__[iclosest]
def closestPointToPath(path : SimplePathHelper, p_query : torch.Tensor, s0 : Union[None, torch.Tensor] = None, lr = 1.0, max_iter = 10000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if s0 is None:
        snaive, _ = closestPointToPathNaive(path, p_query)
        s_optim : torch.nn.parameter.Parameter = torch.nn.parameter.Parameter(torch.as_tensor([snaive], dtype=path.__r_samp__.dtype, device=path.__r_samp__.device), requires_grad=True)
    else:
        s_optim : torch.nn.parameter.Parameter = torch.nn.parameter.Parameter(torch.as_tensor([s0%path.__curve__.xend_vec[-1]], dtype=path.__r_samp__.dtype, device=path.__r_samp__.device)%path.__curve__.xend_vec[-1], requires_grad=True)
    sgd = torch.optim.SGD([s_optim], lr)
    lossprev : torch.Tensor = None
    loss : torch.Tensor = None
    for asdf in range(max_iter):
        x_curr, _ = path(s_optim)
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
    return s_optim.detach()[0], x_curr.detach()
