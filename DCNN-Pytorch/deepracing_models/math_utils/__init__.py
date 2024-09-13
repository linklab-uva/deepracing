from typing import List, Tuple, Union
import typing
from .bezier import bezierLsqfit
from .bezier import bezierM
from .bezier import bezierArcLength
from .bezier import bezierPolyRoots
from .bezier import bezierArcLength, compositeBezierSpline, compositeBezierAntiderivative, compositeBezierFit
from .bezier import closedPathAsBezierSpline, polynomialFormConversion, elevateBezierOrder, compositeBezierEval
from .fitting import pinv, fitAffine
from .bezier import comb

from .statistics import cov
from .integrate import cumtrapz, simpson

from .geometry import localRacelines

from . import bezier


import torch


import torch.nn
import torchaudio
import scipy.spatial
import numpy as np


class CompositeBezierCurve(torch.nn.Module):
    def __init__(self, x : torch.Tensor, control_points : torch.Tensor) -> None:
        super(CompositeBezierCurve, self).__init__()

        self.control_points : torch.nn.Parameter =  torch.nn.Parameter(control_points, requires_grad=False)


        dx = x[1:] - x[:-1]
        if not torch.all(dx>0):
            raise ValueError("x values must be in ascending order")
        
        self.dx : torch.nn.Parameter =  torch.nn.Parameter(dx, requires_grad=False)
        self.xstart_vec : torch.nn.Parameter =  torch.nn.Parameter(x[:-1], requires_grad=False)
        self.xend_vec : torch.nn.Parameter =  torch.nn.Parameter(x[1:], requires_grad=False)

        self.x : torch.nn.Parameter = torch.nn.Parameter(x, requires_grad=False)

        self.d : torch.nn.Parameter = torch.nn.Parameter(torch.as_tensor(self.control_points.shape[-1], dtype=torch.int64), requires_grad=False)

        self.bezier_order : torch.nn.Parameter = torch.nn.Parameter(torch.as_tensor(self.control_points.shape[-2]-1, dtype=torch.int64), requires_grad=False)


    @staticmethod
    def from_file(filepath : str, dtype=torch.float64, device=torch.device("cpu")):
        with open(filepath, "rb") as f:
            statedict = torch.load(f, map_location=device)
        control_points_shape = statedict["control_points"].shape
        fake_x = torch.linspace(0.0, float(control_points_shape[0]+1), steps=control_points_shape[0]+1)
        fake_control_points = torch.zeros(control_points_shape)
        rtn : CompositeBezierCurve = CompositeBezierCurve(fake_x, fake_control_points).to(device=device, dtype=dtype)
        rtn.load_state_dict(statedict)
        return rtn
    
    def forward(self, x_eval : torch.Tensor, idxbuckets : typing.Union[None,torch.Tensor] = None):
        x_true = (x_eval).view(1,-1)
        # x_true = (x_eval%self.xend_vec[-1]).view(1,-1)
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
        control_point_deltas : torch.Tensor = self.bezier_order.detach()*(control_points_detached[:,1:] - control_points_detached[:,:-1])/self.dx.detach()[:,None,None]
        return CompositeBezierCurve(self.x.detach().clone(), control_point_deltas)
class ProbabilisticCBC(torch.nn.Module):

    def __init__(self, x : torch.Tensor, control_points : torch.Tensor, covar_curves : torch.Tensor) -> None:

        super(ProbabilisticCBC, self).__init__()
        self.cbc : CompositeBezierCurve = CompositeBezierCurve(x, control_points)
        numcurves = control_points.shape[0]
        if not (covar_curves.shape[0]==numcurves):
            raise ValueError("Got control points for %d curves but only %d covariances" % (numcurves, covar_curves.shape[0]))
        kbezier = self.cbc.bezier_order.item()
        if not (covar_curves.shape[1] == (kbezier+1)):
            raise ValueError("PCBC of order %d should have %d covariances on each curve, but received %d" % (kbezier, kbezier+1, covar_curves.shape[1])) 
        ambient_dim = control_points.shape[-1]
        if (not (covar_curves.shape[2] == ambient_dim)) or (not (covar_curves.shape[3] == ambient_dim)):
            raise ValueError(("Invalid dimensionality of covariance matrices. Final two dimensions must match ambient dimension of curve. Got ambient "+
                              "dimension %d but covariance matrices have final two dimensions %d x %d") % (ambient_dim, covar_curves.shape[2], covar_curves.shape[3]))
        if not torch.all(covar_curves[:-1,-1]==covar_curves[1:,0]):
            raise ValueError("Final covariance of each curve should be exactly equal to the first covariance of the next curve")
        self.covar_curves : torch.nn.Parameter = torch.nn.Parameter(covar_curves, requires_grad=False)

    def forward(self, t : torch.Tensor, deriv=False):
        positions, idxbuckets = self.cbc(t)
        covar_curves = self.covar_curves[idxbuckets]
        tstart = self.cbc.xstart_vec[idxbuckets]
        tend = self.cbc.xend_vec[idxbuckets] 
        s = (t - tstart)/(tend - tstart)
        M = bezierM(s[:,None], self.cbc.bezier_order).squeeze(-2)
        Msquare = torch.square(M)
        covar_points = torch.sum(Msquare[...,None,None]*covar_curves, dim=-3)
        return positions, covar_points, idxbuckets


    
class SimplePathHelper(torch.nn.Module):
    def __init__(self, arclengths : torch.Tensor, curve_control_points : torch.Tensor, dr_samp : float, leafsize=25) -> None:
        super(SimplePathHelper, self).__init__()
        self.__arclengths_in__ : torch.nn.Parameter = torch.nn.Parameter(arclengths.clone(), requires_grad=False)

        self.__curve__ : CompositeBezierCurve = CompositeBezierCurve(arclengths, curve_control_points).requires_grad_(False)
        self.__curve_deriv__ : CompositeBezierCurve = self.__curve__.derivative().requires_grad_(False)
        self.__curve_2nd_deriv__ : CompositeBezierCurve = self.__curve_deriv__.derivative().requires_grad_(False)

        self.__r_samp__ : torch.nn.Parameter = torch.nn.Parameter(torch.arange(arclengths[0], arclengths[-1], step=dr_samp, dtype=arclengths.dtype, device=arclengths.device), requires_grad=False)
        
        tup : tuple[torch.Tensor, torch.Tensor] = self.__curve__(self.__r_samp__)
        points_samp = tup[0].detach().clone()
        self.__points_samp__ : torch.nn.Parameter = torch.nn.Parameter(points_samp, requires_grad=False)

        self.kd_tree : scipy.spatial.KDTree = scipy.spatial.KDTree(self.__points_samp__.cpu().numpy().astype(np.float32), leafsize=leafsize)

        tup : tuple[torch.Tensor, torch.Tensor] = self.__curve_deriv__(self.__r_samp__)
        tangents_samp = tup[0].detach().clone()
        tangents_samp = tangents_samp/torch.norm(tangents_samp, p=2.0, dim=-1, keepdim=True)
        self.__tangents_samp__ : torch.nn.Parameter = torch.nn.Parameter(tangents_samp, requires_grad=False)

        normals_samp = tangents_samp[:,[1,0]].clone()
        normals_samp[:,0]*=-1.0
        self.__normals_samp__ : torch.nn.Parameter = torch.nn.Parameter(normals_samp, requires_grad=False)
    @staticmethod
    def from_closed_path(points : torch.Tensor, dr_samp : float, leafsize=25) -> 'SimplePathHelper':
        arclengths_, curve_control_points_ = closedPathAsBezierSpline(points)
        return SimplePathHelper(arclengths_, curve_control_points_, dr_samp, leafsize=leafsize)


    def control_points(self):
        return self.__curve__.control_points.detach().clone()
    def r_in(self):
        return self.__arclengths_in__.detach().clone()
    def r_samp(self):
        return self.__r_samp__.detach().clone()
    def normals_samp(self):
        return self.__normals_samp__.detach().clone()
    def tangents_samp(self):
        return self.__tangents_samp__.detach().clone()
    def points_samp(self):
        return self.__points_samp__.detach().clone()
    def offset_points(self, left_offset : float, right_offset: float) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.__points_samp__ + self.__normals_samp__*left_offset, self.__points_samp__ - self.__normals_samp__*right_offset
    def tangent(self, s : torch.Tensor):
        s_true = s%self.__curve_deriv__.xend_vec[-1]
        derivs, _ = self.__curve_deriv__(s_true)
        return derivs
    def forward(self, s : torch.Tensor, deriv=False, idxbuckets=None):
        s_true = s%self.__curve__.xend_vec[-1]
        positions, idxbuckets = self.__curve__(s_true, idxbuckets=idxbuckets)
        if deriv:
            derivs, _ = self.__curve_deriv__(s_true, idxbuckets=idxbuckets)
            return positions, derivs, idxbuckets
        return positions, None, idxbuckets
    def closest_point_approximate(self, Pquery : torch.Tensor,
                newton_iterations = 0, newton_stepsize = 2.0, max_step=1.5, 
                newton_termination_eps : float | None = 5E-4, newton_termination_delta_eps : float | None = 5E-3):
        Pquery_flat = Pquery.view(-1, Pquery.shape[-1])
        imin = torch.as_tensor(self.kd_tree.query(Pquery_flat.cpu().numpy())[1])#, device=Pquery.device)
        if newton_iterations<=0:
            return self.__r_samp__[imin].view(Pquery.shape[:-1]), self.__points_samp__[imin].view(Pquery.shape), self.__tangents_samp__[imin].view(Pquery.shape), self.__normals_samp__[imin].view(Pquery.shape)

        r = self.__r_samp__[imin].clone()
        for _ in range(newton_iterations):
            points, idxbuckets = self.__curve__(r)
            tangents, _ = self.__curve_deriv__(r, idxbuckets=idxbuckets)
            tangents : torch.Tensor = tangents/torch.norm(tangents, p=2.0, dim=-1, keepdim=True)
            dtangent_dr , _  = self.__curve_2nd_deriv__(r, idxbuckets=idxbuckets)
            deltas = Pquery_flat - points
            delta_dotprods = torch.sum(deltas*tangents,dim=-1)
            ddelta_dr = -tangents
            ddotprod_dr = deltas[:,0]*dtangent_dr[:,0] + tangents[:,0]*ddelta_dr[:,0] + deltas[:,1]*dtangent_dr[:,1] + tangents[:,1]*ddelta_dr[:,1]
            newton_step = (delta_dotprods/(2.0*ddotprod_dr))
            r-=(newton_stepsize*newton_step).clip(-max_step, max_step)
            normals : torch.Tensor = tangents[:,[1,0]].clone()
            normals[:,0]*=-1.0
            if torch.all(torch.abs(delta_dotprods)<newton_termination_eps):
                break
            if torch.all(torch.abs(newton_step)<newton_termination_delta_eps):
                break
        return r.view(Pquery.shape[:-1]), points.view(Pquery.shape), tangents.view(Pquery.shape), normals.view(Pquery.shape)
    
    def closest_point(self, Pquery : torch.Tensor):
        order_this = self.__curve__.bezier_order.item()
        order_deriv = order_this - 1
        order_prod = order_this + order_deriv

        control_points = self.__curve__.control_points
        control_points_deriv = self.__curve_deriv__.control_points
        batchdim = Pquery.shape[0]
        control_point_0 = control_points[:,0]
        distance_matrix = torch.cdist(Pquery, control_point_0)
        idx_min = torch.argmin(distance_matrix, dim=1, keepdim=True)
        idx_delta = torch.arange(-5, 6, step=1, dtype=torch.int64, device=Pquery.device)
        idx_delta_exp = (idx_delta.unsqueeze(0).expand(Pquery.shape[0], idx_delta.shape[0]) + idx_min)%control_point_0.shape[0]

        control_points_select = control_points[idx_delta_exp]
        arclengths_start_select = self.__curve__.x[idx_delta_exp.view(-1)].view(Pquery.shape[0], idx_delta_exp.shape[1])
        delta_arclengths_select = self.__curve__.dx[idx_delta_exp.view(-1)].view(Pquery.shape[0], idx_delta_exp.shape[1])
        
        control_points_delta = control_points_select - Pquery[:,None,None]
        control_points_deriv_select = control_points_deriv[idx_delta_exp]

        binomial_coefs = torch.as_tensor([comb(order_this, i) for i in range(order_this+1)], dtype=Pquery.dtype, device=Pquery.device)
        binomial_coefs_deriv = torch.as_tensor([comb(order_deriv, i) for i in range(order_deriv+1)], dtype=Pquery.dtype, device=Pquery.device)
        control_points_delta_scaled = control_points_delta*binomial_coefs[None,None,:,None]
        control_points_deriv_scaled = control_points_deriv_select*binomial_coefs_deriv[None,None,:,None]
    
        convolution = torchaudio.functional.convolve(control_points_delta_scaled.transpose(-2,-1), control_points_deriv_scaled.transpose(-2,-1)).transpose(-2,-1)
        binomial_coefs_prod = torch.as_tensor([comb(order_prod, i) for i in range(order_prod+1)], dtype=Pquery.dtype, device=Pquery.device)
        bezier_polys = torch.sum(convolution/binomial_coefs_prod[None,None,:,None], dim=-1)
        polynom_roots = bezierPolyRoots(bezier_polys.view(-1, order_prod+1)).view(Pquery.shape[0], idx_delta.shape[0], order_prod)
        matchmask = (torch.abs(polynom_roots.imag)<1E-5)*(polynom_roots.real>0.0)*(polynom_roots.real<1.0)
        idx=torch.arange(0, idx_delta.shape[0], step=1, dtype=torch.int64, device=control_points.device)
        selection_all = (torch.sum(matchmask, dim=-1)>=1)
        rclosest = torch.empty_like(Pquery[:,0])

        for i in range(batchdim):
            selection = selection_all[i]
            candidates_idx = idx[selection]
            candidates = control_points_select[i,candidates_idx]
            candidates_polyroots = polynom_roots[i,candidates_idx]
            
            candidates_rstart = arclengths_start_select[i,candidates_idx]
            candidates_dr = delta_arclengths_select[i,candidates_idx]
            
            norms = torch.norm(candidates[:,[0,-1]], p=2.0, dim=2)
            norm_means = torch.mean(norms, dim=1)
            imin = torch.argmin(norm_means)

            correctroots = candidates_polyroots[imin]
            correctsval = correctroots[(torch.abs(correctroots.imag)<1E-6)*(correctroots.real>=0.0)*(correctroots.real<=1.0)].real.item()
            correctdr = candidates_dr[imin]

            correctrstart = candidates_rstart[imin]

            rclosest[i] = correctrstart + correctsval*correctdr

        return rclosest






    def y_axis_intersection(self, Pquery : torch.Tensor, Rquery : torch.Tensor):
        
        control_points = self.__curve__.control_points

        batchdim = Pquery.shape[0]
        control_point_0 = control_points[:,0]
        distance_matrix = torch.cdist(Pquery, control_point_0)
        idx_min = torch.argmin(distance_matrix, dim=1, keepdim=True)
        idx_delta = torch.arange(-15, 16, step=1, dtype=torch.int64, device=Pquery.device)
        idx_delta_exp = (idx_delta.unsqueeze(0).expand(Pquery.shape[0], idx_delta.shape[0]) + idx_min)%control_point_0.shape[0]

        control_points_select = control_points[idx_delta_exp]#.view(-1)].view(Pquery.shape[0], idx_delta_exp.shape[1], control_points.shape[-2], control_points.shape[-1])

        # arclengths_start = self.__curve__.x
        # delta_arclengths = self.__curve__.dx

        arclengths_start_select = self.__curve__.x[idx_delta_exp.view(-1)].view(Pquery.shape[0], idx_delta_exp.shape[1])
        delta_arclengths_select = self.__curve__.dx[idx_delta_exp.view(-1)].view(Pquery.shape[0], idx_delta_exp.shape[1])

        
        rotmat = Rquery.transpose(-2,-1).to(control_points.device).to(control_points.dtype)
        pquery = Pquery.to(control_points.device).to(control_points.dtype)
        ptransform = -torch.matmul(rotmat, pquery.unsqueeze(-1)).squeeze(-1)

        control_points_exp = control_points_select

        control_points_exp_flat = control_points_exp.view(batchdim, -1, control_points.shape[-1])
        control_points_transformed_flat = torch.matmul(rotmat, control_points_exp_flat.transpose(-2,-1)).transpose(-2,-1) + ptransform[:,None]

        control_points_transformed = control_points_transformed_flat.view(control_points_exp.shape)


        # xbezier_flat = control_points_transformed[:,:,:,0].reshape(-1, control_points.shape[-2])
        # polynom_roots = bezierPolyRoots(xbezier_flat).view(batchdim, control_points_exp.shape[1], control_points.shape[-2] - 1)
        # polynom_roots = torch.stack([ for i in range(batchdim)], dim=0).view(batchdim, -1, 3)
        

        idx=torch.arange(0, control_points_exp.shape[1], step=1, dtype=torch.int64, device=control_points.device)
        
        rintersect = torch.empty_like(pquery[:,0])


        for i in range(batchdim):
            current_points_transformed = control_points_transformed[i]#.cpu().numpy()
            xpolys = current_points_transformed[:,:,0]
            polynom_roots = bezierPolyRoots(xpolys)
            current_roots_real = polynom_roots.real #.cpu().numpy()
            current_roots_imag = polynom_roots.imag #.cpu().numpy()
            matchmask = (torch.abs(current_roots_imag)<1E-8)*(current_roots_real>=0.0)*(current_roots_real<=1.0)
            selection = torch.sum(matchmask, dim=-1)>=1
            candidates_idx = idx[selection]
            candidates = current_points_transformed[candidates_idx]
            candidates_polyroots = polynom_roots[candidates_idx]

            candidates_rstart = arclengths_start_select[i,candidates_idx]
            candidates_dr = delta_arclengths_select[i,candidates_idx]
            
            norms = torch.norm(candidates[:,[0,-1]], p=2.0, dim=2)
            norm_means = torch.mean(norms, dim=1)
            imin = torch.argmin(norm_means)
            # if norm_means[imin]>20.0:
            #     current_pquery = Pquery[i]
            #     pass
            correctroots = candidates_polyroots[imin]
            correctsval = correctroots[(torch.abs(correctroots.imag)<1E-8)*(correctroots.real>=0.0)*(correctroots.real<=1.0)].real.item()
            correctdr = candidates_dr[imin]


            correctrstart = candidates_rstart[imin]

            rintersect[i] = correctrstart + correctsval*correctdr

        return rintersect
    
    

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
