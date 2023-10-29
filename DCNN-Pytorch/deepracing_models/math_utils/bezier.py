from typing import Tuple, Union
import numpy as np
import math
import torch, torch.nn
from deepracing_models.math_utils.fitting import pinv
from ..math_utils.polynomial import polyroots
import torch.jit
import typing
import functools


def compositeBezierSpline(x : torch.Tensor, Y : torch.Tensor, boundary_conditions : Union[str,torch.Tensor] = "periodic"):
    if boundary_conditions=="periodic":
        return compositeBezierSpline_periodic_(x,Y)
    elif type(boundary_conditions)==torch.Tensor:
        return compositeBezierSpline_with_boundary_conditions_(x, Y, boundary_conditions)
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
    distances = bezierArcLength(euclidean_spline)
    arclengths[1:] = torch.cumsum(distances, 0)
    return arclengths, compositeBezierSpline(arclengths,Yaug,boundary_conditions="periodic")



def compositeBezierAntiderivative(control_points : torch.Tensor, delta_t : torch.Tensor | float) -> torch.Tensor:
    numsplinesegments : int = control_points.shape[-3]
    kbezier_out : int = control_points.shape[-2]
    d : int = control_points.shape[-1]
    shapeout : list[int] = list(control_points.shape)
    shapeout[-2]+=1

    control_points_onebatchdim = control_points.view(-1, numsplinesegments, kbezier_out, d)
    batchdim = control_points_onebatchdim.shape[0]
    if type(delta_t) is float:
        delta_t_onebatchdim = delta_t*torch.ones([batchdim, numsplinesegments], dtype=control_points.dtype, device=control_points.device)
    else:
        delta_t_onebatchdim = delta_t.view(-1, numsplinesegments)

    antiderivative_onebatchdim = torch.zeros(batchdim, numsplinesegments, kbezier_out+1, d, dtype=control_points.dtype, device=control_points.device)
    antiderivative_onebatchdim[:,0,1:] += delta_t_onebatchdim[:,0,None,None]*torch.cumsum(control_points_onebatchdim[:,0], dim=1)
    for seg in range(1,numsplinesegments):
        antiderivative_onebatchdim[:, seg] += (antiderivative_onebatchdim[:, seg-1, -1])[:,None]
        antiderivative_onebatchdim[:, seg, 1:] += delta_t_onebatchdim[:,seg,None,None]*torch.cumsum(control_points_onebatchdim[:,seg], dim=1)   
    
    return (antiderivative_onebatchdim).view(shapeout)/kbezier_out

def compositeBezierFit(x : torch.Tensor, points : torch.Tensor, numsegments : int, 
                       kbezier : int = 3, dYdT_0 : torch.Tensor | None = None, dYdT_f : torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = points.dtype
    device = points.device
    batchdims = list(points.shape[:-2])
    num_points = points.shape[-2]
    numcoefs = kbezier+1
    dim = points.shape[-1]

    x = x.view(-1, num_points)
    points = points.view(-1, num_points, dim)  
    if dYdT_0 is not None:
        dYdT_0 = dYdT_0.view(-1, dim)
    if dYdT_f is not None:
        dYdT_f = dYdT_f.view(-1, dim)
    batchdimflat = points.shape[0]          
        
    if not torch.all((x[:, 1:] - x[:, :-1])>0.0):
        raise ValueError("x must be monotonically increasing")
    tsamp = x.clone()

    constrain_initial_derivative = dYdT_0 is not None
    constrain_final_derivative = dYdT_f is not None
    continuinty_constraits_per_segment = min(kbezier, 3)
    continuity_constraints = int(continuinty_constraits_per_segment*(numsegments-1))
    total_constraints = continuity_constraints + 2 + int(constrain_initial_derivative) + int(constrain_final_derivative)
    
    tswitchingpoints = torch.stack([
        torch.linspace(tsamp[i,0].item(), tsamp[i,-1].item(), steps=numsegments+1, dtype=dtype, device=device)
        for i in range(batchdimflat)
        ], 
    dim=0)
    
    tstart = (tswitchingpoints[:, :-1])
    tend = (tswitchingpoints[:, 1:])
    dt = tend - tstart

    print("Building linear system")
    HugeM : torch.Tensor = torch.zeros([
        batchdimflat,  num_points, numcoefs*numsegments
        ], device=device, dtype=dtype)
    #come back to the autograd stuff later
    # def fakeEval(idx, points):
    #     return compositeBezierEval(tstart[idx], dt[idx], points, x[idx])[0].squeeze(-1)
    # placeholder_control_points=torch.ones_like(dt[0,:,None,None].expand(numsegments, kbezier+1, 1))
    # for b in range(batchdimflat):
    #     partial = functools.partial(fakeEval, b)
    #     autograd_result = torch.autograd.functional.jacobian(partial, placeholder_control_points)
    #     HugeM[b] = autograd_result.view(-1, numsegments*(kbezier+1))
    for b in range(batchdimflat):
        curr_switchpoints = tswitchingpoints[b]
        curr_tsamp = tsamp[b]
        curr_tstart = tstart[b]
        curr_dt = dt[b]
        idxbucket = torch.bucketize(curr_tsamp, curr_switchpoints, right=True) - 1
        segment_sizes = []
        for i in range(numsegments):
            idxselect = idxbucket==i
            subt = curr_tsamp[idxselect]
            subs = ((subt - curr_tstart[i])/curr_dt[i])
            column_start = i*numcoefs
            column_end =  column_start + numcoefs
            HugeM[b, idxselect, column_start:column_end] = \
                bezierM(subs.unsqueeze(0), kbezier)[0]
            segment_sizes.append(torch.sum(idxselect).item())
    print("Solving linear system")
    Q = torch.matmul(HugeM.transpose(-2, -1), HugeM)
    E = torch.zeros(batchdimflat, total_constraints, Q.shape[-1], dtype=Q.dtype, device=Q.device)
    d = torch.zeros(batchdimflat, total_constraints, dim, dtype=Q.dtype, device=Q.device)
    if continuinty_constraits_per_segment>=1:
        for i in range(int(continuity_constraints/continuinty_constraits_per_segment)):
            E[:, i, (i+1)*(kbezier+1)-1] = -1.0
            E[:, i, (i+1)*(kbezier+1)] = 1.0
    if continuinty_constraits_per_segment>=2:
        for i in range(int(continuity_constraints/continuinty_constraits_per_segment)):
            row = i + int(continuity_constraints/continuinty_constraits_per_segment)
            E[:, row, (i+1)*(kbezier+1)-2] = -1.0
            E[:, row, (i+1)*(kbezier+1)-1] = 1.0
            E[:, row, (i+1)*(kbezier+1)] = 1.0
            E[:, row, (i+1)*(kbezier+1)+1] = -1.0
    if continuinty_constraits_per_segment>=3:
        for i in range(int(continuity_constraints/continuinty_constraits_per_segment)):
            row = i + 2*int(continuity_constraints/continuinty_constraits_per_segment)
            E[:, row, (i+1)*(kbezier+1)-3] = 1.0
            E[:, row, (i+1)*(kbezier+1)-2] = -2.0
            E[:, row, (i+1)*(kbezier+1)-1] = 1.0

            E[:, row, (i+1)*(kbezier+1)] = -1.0
            E[:, row, (i+1)*(kbezier+1)+1] = 2.0
            E[:, row, (i+1)*(kbezier+1)+2] = -1.0
    E[:, continuity_constraints,0] = 1.0
    d[:, continuity_constraints] = points[:, 0]
    E[:, continuity_constraints+1,-1] = 1.0
    d[:, continuity_constraints+1] = points[:, -1]
    if constrain_initial_derivative:
        E[:, continuity_constraints+2,0] = -kbezier
        E[:, continuity_constraints+2,1] = kbezier
        d[:, continuity_constraints+2] = dYdT_0*dt[:, 0, None]
    if constrain_final_derivative:
        E[:, continuity_constraints + 2 + int(constrain_initial_derivative),-2] = -kbezier
        E[:, continuity_constraints + 2 + int(constrain_initial_derivative),-1] = kbezier
        d[:, continuity_constraints + 2 + int(constrain_initial_derivative)] = dYdT_f*dt[:, -1, None]


    lhs = torch.zeros([batchdimflat, Q.shape[1] + E.shape[1], Q.shape[1] + E.shape[1]], dtype=Q.dtype, device=Q.device)
    lhs[:, :Q.shape[1],:Q.shape[1]] = Q
    lhs[:, Q.shape[1]:,:E.shape[2]] = E
    lhs[:, :E.shape[2], Q.shape[1]:] = E.transpose(-2, -1)
    rhs = torch.cat([torch.matmul(HugeM.transpose(-2, -1), points), d], dim=1)

    coefs_and_lagrange = torch.linalg.solve(lhs, rhs)
    coefs = coefs_and_lagrange[:, :-d.shape[1]]
    control_points =  coefs.view(batchdims + [numsegments, numcoefs, dim])
    tswitchingpoints_batch = tswitchingpoints.view(batchdims + [numsegments + 1,])
    # lagrange = coefs_and_lagrange[:, d.shape[1]:]
    # lagrange_batch = lagrange.reshape(batchdims + [-1,])
    return control_points, tswitchingpoints_batch

    
def compositeBezierEval(xstart : torch.Tensor, dx : torch.Tensor, 
                        control_points : torch.Tensor, x_eval : torch.Tensor, 
                        idxbuckets : typing.Union[torch.Tensor,None] = None
                        ) -> typing.Tuple[torch.Tensor, torch.Tensor]:

    numpoints : int = x_eval.shape[-1]
    numsplinesegments : int = control_points.shape[-3]
    kbezier : int = control_points.shape[-2] - 1
    d : int = control_points.shape[-1]

    xstart_onebatchdim : torch.Tensor = xstart.view(-1, numsplinesegments)
    batchsize : int = xstart_onebatchdim.shape[0]
    x_eval_onebatchdim : torch.Tensor = x_eval.view(batchsize, numpoints)
    control_points_onebatchdim : torch.Tensor = control_points.view(batchsize, numsplinesegments, kbezier+1, d)
    dx_onebatchdim : torch.Tensor = dx.view(batchsize, numsplinesegments)


    if idxbuckets is None:
        if batchsize == 1:
            idxbuckets_ : torch.Tensor = (torch.bucketize(x_eval_onebatchdim[0], xstart_onebatchdim[0], right=True).view(1, numpoints) - 1).clip(min=0)
        else:
            idxbuckets_ : torch.Tensor = (torch.stack([torch.bucketize(x_eval_onebatchdim[i], xstart_onebatchdim[i], right=True) for i in range(batchsize)], dim=0) - 1).clip(min=0)
    else:
        idxbuckets_ : torch.Tensor = idxbuckets.view(batchsize, numpoints).clip(min=0)
        
    idxbuckets_exp = idxbuckets_.unsqueeze(-1).unsqueeze(-1).expand(batchsize, numpoints, kbezier+1, d)#%xstart_onebatchdim.shape[1]
    corresponding_curves = torch.gather(control_points_onebatchdim, 1, idxbuckets_exp)
    corresponding_xstart = torch.gather(xstart_onebatchdim, 1, idxbuckets_)
    corresponding_dx = torch.gather(dx_onebatchdim, 1, idxbuckets_)
    s_eval = (x_eval_onebatchdim - corresponding_xstart)/corresponding_dx
    s_eval_unsqueeze = s_eval.unsqueeze(-1)
    Mbezier = bezierM(s_eval_unsqueeze.view(-1, 1), kbezier).view(batchsize, numpoints, kbezier+1)
    pointseval = torch.matmul(Mbezier.unsqueeze(-2), corresponding_curves).squeeze(-2)
    idxbuckets_shape_out = x_eval.shape 
    if d>1:
        points_shape_out = list(idxbuckets_shape_out) + [d]
    else:
        points_shape_out = idxbuckets_shape_out
    return pointseval.view(points_shape_out), idxbuckets_.view(idxbuckets_shape_out)

def polynomialFormConversion(k : int, dtype=torch.float64, device=torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
    topolyform : torch.Tensor = torch.zeros((k+1,k+1), dtype=dtype, device=device)
    kfactorial : float = math.factorial(k)
    topolyform[0,0]=topolyform[-1,-1]=1.0
    for i in range(1, k+1):
        outerfactor = kfactorial/math.factorial(k-i)
        for j in range(0,i+1):
            topolyform[i,j]=outerfactor/(math.factorial(j)*math.factorial(i-j))
            if ((i+j)%2)!=0:
                topolyform[i,j]*=-1.0
    #inverse of a lower triangular matrix is also lower triangular.
    #force elements above the main diagonal to be zero
    tobezierform = torch.linalg.inv(topolyform)
    for i in range(0,k+1):
        for j in range(0,k+1):
            if j>i:
                tobezierform[i,j]=0.0
    return topolyform, tobezierform

from scipy.special import comb

def bezierPolyRoots(bezier_coefficients : torch.Tensor, scaled_basis = False):
    N = bezier_coefficients.shape[0]
    k = bezier_coefficients.shape[1]-1
    topolyform, _ = polynomialFormConversion(k, dtype = bezier_coefficients.dtype, device=bezier_coefficients.device)
    topolyform = topolyform.unsqueeze(0).expand(N, k+1, k+1)
    if scaled_basis:
        binoms = torch.as_tensor([comb(k, i, exact=True) for i in range(k+1)], dtype = bezier_coefficients.dtype, device=bezier_coefficients.device)
        unscaled = bezier_coefficients/binoms
        standard_form = torch.matmul(topolyform, unscaled.unsqueeze(-1)).squeeze(-1)
    else:
        standard_form = torch.matmul(topolyform, bezier_coefficients.unsqueeze(-1)).squeeze(-1)
    return polyroots(standard_form)


def Mtk(k : int, n : int, t : torch.Tensor, scaled_basis=False):
    rtn = torch.pow(t,k)*torch.pow(1-t,(n-k))
    if scaled_basis:
        return rtn
    else:
        return rtn*comb(n, k, exact=True)

from scipy.special import roots_legendre
def bezierArcLength(control_points : torch.Tensor, quadrature_order = 7, num_segments = 4, sum=True):
    
    batchdim = control_points.shape[0]
    kbezier = control_points.shape[-2] - 1
    dim = control_points.shape[-1]
    
    dtype = control_points.dtype
    device = control_points.device

    ds = 1.0/num_segments
    
    a = torch.linspace(0.0, 1.0 - ds, steps=num_segments, dtype=dtype, device=device)
    b = a + ds

    roots_np, weights_np = roots_legendre(quadrature_order)
    roots = torch.as_tensor(roots_np, dtype=dtype, device=device)
    weights = torch.as_tensor(weights_np, dtype=dtype, device=device)

    s_shifted = 0.5*ds*roots.unsqueeze(0).expand(num_segments, quadrature_order) + 0.5*(a+b)[:,None]

    bezierMderiv = bezierM(s_shifted, kbezier-1)
    bezierMderiv_exp = bezierMderiv.unsqueeze(0).expand(batchdim, num_segments, quadrature_order, kbezier)

    control_points_deriv = kbezier*(control_points[:,1:] - control_points[:,:-1])
    control_points_deriv_exp = control_points_deriv.unsqueeze(1).expand(batchdim, num_segments, kbezier, dim)

    quadrature_node_values = torch.matmul(bezierMderiv_exp, control_points_deriv_exp)
    quadrature_node_norms = torch.norm(quadrature_node_values, p=2.0, dim=-1)

    segment_sums = 0.5*ds*torch.sum(quadrature_node_norms*weights[None,None], dim=-1)

    if sum:
        return torch.sum(segment_sums, dim=1)
    
    return segment_sums


def compositeBezierSpline_with_boundary_conditions_(x : torch.Tensor, Y : torch.Tensor, boundary_conditions : torch.Tensor):
    
    k=3
    if x.shape[0]!=Y.shape[0]:
        raise ValueError("x and Y must be equal in size in dimension 0. x.shape[0]=%d, but Y.shape[0]=%d" % (x.shape[0], Y.shape[0])) 
    lhs1, rhs1 = first_order_constraints(x, Y, k=k, bc_type = boundary_conditions)
    lhs2, rhs2 = second_order_constraints(x, Y, k=k, periodic = False)
    lhs = torch.cat([lhs1, lhs2], dim=0)
    rhs = torch.cat([rhs1, rhs2], dim=0)
    solution = torch.linalg.solve(lhs, rhs)
    intermediate_points = solution.view(Y.shape[-2] - 1, 2, -1)
    return torch.cat([Y[:-1].unsqueeze(1), intermediate_points, Y[1:].unsqueeze(1)], dim=1)        

def first_order_constraints(x : torch.Tensor, Y : torch.Tensor, bc_type : torch.Tensor | str = "periodic", k=3):
    #Eventually, i'll get around to implementing non-cubic splines, but for now this works.
    k=3
    if not x.shape[-1]==Y.shape[-2]:
        raise ValueError("x must have same number of elements as Y has rows")
    d = Y.shape[-1]
    num_points = Y.shape[-2]
    num_segments = num_points - 1
    batchdims = list(x.shape[:-1])
    x = x.view(-1, num_points)
    Y = Y.view(-1, num_points, d)
    batchdimflat = Y.shape[0]
    deltat = x[:, 1:] - x[:, :-1]
    kappa = 1.0/deltat
    periodic = bc_type=="periodic"
    if periodic:
        num_constraints = num_segments
        rhs : torch.Tensor = Y[:, 1:].clone()
        # isegs = (np.arange(1, num_segments + 1, step=1, dtype=np.int64) % num_segments).tolist()
        isegs = torch.arange(1, num_segments + 1, step=1, dtype=torch.int64) % num_segments
    else:
        bc_type = bc_type.view(batchdimflat, 2, d)
        num_constraints = num_segments + 1
        rhs : torch.Tensor = torch.zeros([batchdimflat, num_constraints, d], dtype=Y.dtype, device=Y.device)
        rhs[:, :-2] = Y[:, 1:-1].clone()
        # isegs = np.arange(1, num_segments, step=1, dtype=np.int64).tolist()
        isegs = torch.arange(1, num_segments, step=1, dtype=torch.int64)

    lhs : torch.Tensor = torch.zeros([batchdimflat, num_constraints, num_segments, 2], dtype=Y.dtype, device=Y.device)
    for (constraint_idx, point_idx) in enumerate(isegs):
        kappasum = kappa[:, point_idx-1] + kappa[:, point_idx]
        lhs[:, constraint_idx, point_idx-1, 1] = kappa[:, point_idx-1]/kappasum
        lhs[:, constraint_idx, point_idx, 0] = kappa[:, point_idx]/kappasum

    # kappasum = kappa[:, isegs-1] + kappa[:, isegs]
    # lhs[:, :isegs.shape[0], isegs-1, 1] = kappa[:, isegs-1]/kappasum
    # lhs[:, :isegs.shape[0], isegs, 0] = kappa[:, isegs]/kappasum

    if not periodic:
        V0 = bc_type[:, 0]
        lhs[:, -2, 0, 0] = kappa[:, 0]
        rhs[:, -2] = V0/k +  Y[:, 0]*kappa[:, 0]

        Vf = bc_type[:, 1]
        lhs[:, -1, -1, 1] = -kappa[:, -1]
        rhs[:, -1] = Vf/k - Y[:, -1]*kappa[:, -1]

    return lhs.view(batchdims + [num_constraints, 2*num_segments]), rhs.view(batchdims + [num_constraints, d])

def second_order_constraints(x : torch.Tensor, Y : torch.Tensor, periodic = True, k=3):
    if not x.shape[-1]==Y.shape[-2]:
        raise ValueError("x must have same number of elements as Y has rows")
    d = Y.shape[-1]
    num_points = Y.shape[-2]
    num_segments = num_points - 1
    batchdims = list(x.shape[:-1])
    x = x.view(-1, num_points)
    Y = Y.view(-1, num_points, d)
    batchdimflat = Y.shape[0]
    deltat = x[:,1:] - x[:,:-1]
    kappa = 1.0/torch.square(deltat)
    if periodic:
        num_constraints = num_segments
        isegs = np.arange(0, num_constraints, step=1, dtype=np.int64).tolist()
        rhs : torch.Tensor = torch.zeros([batchdimflat, num_constraints, d], dtype=Y.dtype, device=Y.device)
        # lhs : torch.Tensor = torch.zeros([batchdimflat, num_constraints, num_segments, 2], dtype=Y.dtype, device=Y.device)
    else:
        num_constraints = num_segments - 1
        isegs = np.arange(1, num_segments, step=1, dtype=np.int64).tolist()
        rhs : torch.Tensor = torch.zeros([batchdimflat, num_constraints, d], dtype=Y.dtype, device=Y.device)
    lhs : torch.Tensor = torch.zeros([batchdimflat, num_constraints, num_segments, 2], dtype=Y.dtype, device=Y.device)
    for (constraint_idx, point_idx) in enumerate(isegs):
        dkappa = kappa[:,point_idx] - kappa[:,point_idx-1]
        rhs[:,constraint_idx] = dkappa*Y[:,constraint_idx]

        lhs[:,constraint_idx, point_idx-1, 0] = kappa[:,point_idx-1]
        lhs[:,constraint_idx, point_idx-1, 1] = -2.0*kappa[:,point_idx-1]

        lhs[:,constraint_idx, point_idx, 0] = 2.0*kappa[:,point_idx]
        lhs[:,constraint_idx, point_idx, 1] = -kappa[:,point_idx]     
    return lhs.view(batchdims + [num_constraints, 2*num_segments]), rhs.view(batchdims + [num_constraints, d])

def compositeBezierSpline_periodic_(x : torch.Tensor, Y : torch.Tensor):
    k=3
    if torch.linalg.norm(Y[-1] - Y[0], ord=2)>1E-5:
        raise ValueError("Y[-1] and Y[0] must be the same value for a periodic spline")
    if x.shape[0]!=Y.shape[0]:
        raise ValueError("x and Y must be equal in size in dimension 0. x.shape[0]=%d, but Y.shape[0]=%d" % (x.shape[0], Y.shape[0])) 
    lhs1, rhs1 = first_order_constraints(x, Y, k=k, bc_type = "periodic")
    lhs2, rhs2 = second_order_constraints(x, Y, k=k, periodic = True)
    lhs = torch.cat([lhs1, lhs2], dim=0)
    rhs = torch.cat([rhs1, rhs2], dim=0)
    solution = torch.linalg.solve(lhs, rhs)
    intermediate_points = solution.view(Y.shape[-2] - 1, 2, -1)
    return torch.cat([Y[:-1].unsqueeze(1), intermediate_points, Y[1:].unsqueeze(1)], dim=1)

def bezierM(s : torch.Tensor, n : int, scaled_basis : bool = False) -> torch.Tensor:
    return torch.stack([Mtk(k,n,s, scaled_basis=scaled_basis) for k in range(n+1)],dim=2)

def bezierLsqfit(points, n, t = None, M = None, built_in_lstq = False, P0 : torch.Tensor | None = None, V0 : torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if ((t is None) and (M is None)) or ((t is not None) and (M is not None)):
        raise ValueError("One of t or M must be set, but not both")
    if M is None:
        t0 = t[:,0]
        dt = t[:,-1] - t0
        s = (t - t0[:,None])/dt[:,None]
        M_ = bezierM(s,n)
    else:
        M_ = M

    if (P0 is None) and (V0 is None):
        res = torch.linalg.lstsq(M_, points)
        return M_, res.solution
    elif (V0 is not None) and (P0 is None):
        raise ValueError("P0 must be provided to constrain V0")
    
    batch = points.shape[0]
    ambient_dimension = points.shape[2]      
    M_transpose = M_.transpose(-2, -1)

    num_constraints = int(P0 is not None) + int(V0 is not None)

    num_coefs = n + 1
    matdim = num_coefs + num_constraints

    lhs = torch.zeros([batch, matdim, matdim], dtype=M_.dtype, device=M_.device)
    torch.matmul(M_transpose, M_, out=lhs[:, 0 : num_coefs, 0 : num_coefs])
    lhs[:, num_coefs, 0] = lhs[:, 0, num_coefs] = 1.0
    if (V0 is not None):
        lhs[:, -1, 0] = lhs[:, 0, -1] = -1.0
        lhs[:, -1, 1] = lhs[:, 1, -1] = 1.0

    rhs = torch.zeros([batch, matdim, ambient_dimension], dtype=M_.dtype, device=M_.device)
    torch.matmul(M_transpose, points, out=rhs[:,0:num_coefs])
    rhs[:, num_coefs]=P0
    if (V0 is not None):
        rhs[:,-1]=V0/n

    res = torch.linalg.lstsq(lhs, rhs)
    return M_, res.solution[:,0:num_coefs]
    
def elevateBezierOrder(points : torch.Tensor, out : Union[None, torch.Tensor] = None) -> torch.Tensor:
    originalshape : torch.Size = points.size()
    
    d = originalshape[-1]
    k = originalshape[-2] - 1

    points_flat = points.view(-1, k+1, d)
    sizeout = list(originalshape)
    sizeout[-2]+=1
    if out is None:
        out_flat = torch.empty([torch.prod(torch.as_tensor(originalshape[:-2])).item(), k+2, d], device=points.device, dtype=points.dtype)
    else:
        out_flat = out.view(-1 , k+2, d)
    out_flat[:,0] = points_flat[:,0]
    out_flat[:,-1] = points_flat[:,-1]
    batchdim = out_flat.shape[0]
    coefs = torch.linspace(1.0/(k+1), k/(k+1), steps=k, device=points.device, dtype=points.dtype).unsqueeze(0).expand(batchdim, k)
    out_flat[:,1:-1] = points_flat[:,1:]*coefs[:,:,None] + points_flat[:,:-1]*((1.0-coefs)[:,:,None])
    return out_flat.view(sizeout)
    