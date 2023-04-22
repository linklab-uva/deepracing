from .bezier import bezierLsqfit
from .bezier import bezierM
from .bezier import bezierDerivative
from .fitting import pinv, fitAffine
from .bezier import bezierArcLength as bezierArcLength, BezierCurveModule, polynomialFormConversion

from .statistics import cov
from .integrate import cumtrapz, simpson

from .geometry import localRacelines

import deepracing_models.math_utils.bezier

import torch
def compositeBezierSpline(x : torch.Tensor, Y : torch.Tensor, bc_type : str ="periodic"):
    return deepracing_models.math_utils.bezier.compositeBezierSpline_periodic_(x,Y)