from typing import Tuple
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
def compositeBezierSpline(x : torch.Tensor, Y : torch.Tensor, bc_type : str ="periodic"):
    return bezier.compositeBezierSpline_periodic_(x,Y)

def closedPathAsBezierSpline(Y : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    euclidean_distances : torch.Tensor = torch.zeros_like(Y[:,0])
    delta_euclidean_distances = torch.norm( Y[1:]-Y[:-1] , dim=1 )
    euclidean_distances[1:] = torch.cumsum( delta_euclidean_distances, 0 )
    euclidean_spline = bezier.compositeBezierSpline_periodic_(euclidean_distances,Y)
    arclengths = torch.zeros_like(Y[:,0])
    _, _, _, _, distances = bezierArcLength(euclidean_spline, N=5, simpsonintervals=20)
    arclengths[1:] = torch.cumsum(distances[:,-1], 0)
    return arclengths, bezier.compositeBezierSpline_periodic_(arclengths,Y)