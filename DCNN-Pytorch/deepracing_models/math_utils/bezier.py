from typing import Tuple, Union
import numpy as np
import math
import torch, torch.nn
from deepracing_models.math_utils.fitting import pinv
from ..math_utils.polynomial import polyroots
import torch.jit
import typing


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
    _, _, _, _, distances = bezierArcLength(euclidean_spline, N=2, simpsonintervals=20)
    arclengths[1:] = torch.cumsum(distances[:,-1], 0)
    return arclengths, compositeBezierSpline(arclengths,Yaug,boundary_conditions="periodic")

def compositeBezierAntiderivative(control_points : torch.Tensor, delta_t : torch.Tensor,  p0 : typing.Union[torch.Tensor, None] = None) -> torch.Tensor:
    numsplinesegments : int = control_points.shape[-3]
    kbezier_in : int = control_points.shape[-2]-1
    d : int = control_points.shape[-1]
    shapeout : list = list(control_points.shape)
    shapeout[-2]+=1
    control_points_onebatchdim = control_points.view(-1, numsplinesegments, kbezier_in+1, d)
    batchdim = control_points_onebatchdim.shape[0]
    antiderivative_onebatchdim = torch.empty(batchdim, numsplinesegments, kbezier_in+2, d, dtype=control_points.dtype, device=control_points.device)
    if p0 is None:
        antiderivative_onebatchdim[:,0,0]=torch.zeros_like(control_points_onebatchdim[:,0,0])
    else:
       antiderivative_onebatchdim[:,0,0]=p0.view(batchdim, d)
    antiderivative_onebatchdim[:,0,1:] = antiderivative_onebatchdim[:,0,0,None] + torch.cumsum(control_points_onebatchdim[:,0], dim=1)
    for seg in range(1, numsplinesegments):
        antiderivative_onebatchdim[:, seg, 0]  = antiderivative_onebatchdim[:, seg-1, -1]
        antiderivative_onebatchdim[:, seg, 1:] = antiderivative_onebatchdim[:, seg, 0, None] + torch.cumsum(control_points_onebatchdim[:,0], dim=1)   
    return (delta_t[:,:,None,None]*antiderivative_onebatchdim.view(shapeout))/(kbezier_in + 1)

def compositeBezierFit(points : torch.Tensor, t : torch.Tensor, numsegments : int, kbezier : int = 3):
    dtype = points.dtype
    device = points.device
    if not t[0]==0.0:
        raise ValueError("t must be start at 0.0")
    if not torch.all((t[1:] - t[:-1])>0.0):
        raise ValueError("t must be monotonically increasing")
    tsamp = torch.as_tensor(t, dtype=dtype, device=device)

    # vels = torch.as_tensor(raceline_pathhelper_cpu.spline_time_derivative(_tsamp), dtype=tsamp.dtype, device=tsamp.device)
    # speedsamp = torch.norm(vels, p=2, dim=1)
    if not (tsamp.shape[0]%numsegments)==0:
        raise ValueError("Number of fit points, %d, must be divisible by number of segments. %d is not divisible by %d" % (tsamp.shape[0],tsamp.shape[0],numsegments))
    points_per_segment = math.ceil(tsamp.shape[0]/numsegments)
    tsamp_dense = tsamp.view(numsegments, -1)
    # torch.set_printoptions(linewidth=500, precision=3)
    continuinty_constraits_per_segment = min(kbezier, 3)
    continuity_constraints = int(continuinty_constraits_per_segment*(numsegments-1))
    total_constraints = continuity_constraints + 1
    numcoefs = kbezier+1
    HugeM_dense : torch.Tensor = torch.zeros([numsegments, points_per_segment, numsegments, numcoefs], device=device, dtype=dtype)
    tswitchingpoints = torch.linspace(0.0, tsamp[-1].item(), steps=numsegments+1, dtype=dtype, device=device)
    tstart = (tswitchingpoints[:-1]).clone()
    tend = (tswitchingpoints[1:]).clone()
    dt = tend - tstart
    for i in range(numsegments):
        subt = tsamp_dense[i] - tswitchingpoints[i]
        subs = subt/dt[i]
        HugeM_dense[i, :, i] = bezierM(subs.unsqueeze(0), kbezier).squeeze(0)
    HugeM : torch.Tensor = HugeM_dense.view(tsamp.shape[0], numcoefs*numsegments)
    Q = torch.matmul(HugeM.t(), HugeM)
    E = torch.zeros(total_constraints, Q.shape[1], dtype=Q.dtype, device=Q.device)
    d = torch.zeros(total_constraints, points.shape[-1], dtype=Q.dtype, device=Q.device)
    if continuinty_constraits_per_segment>=1:
        for i in range(int(continuity_constraints/continuinty_constraits_per_segment)):
            E[i, (i+1)*(kbezier+1)-1] = -1.0
            E[i, (i+1)*(kbezier+1)] = 1.0
    if continuinty_constraits_per_segment>=2:
        for i in range(int(continuity_constraints/continuinty_constraits_per_segment)):
            row = i + int(continuity_constraints/continuinty_constraits_per_segment)
            E[row, (i+1)*(kbezier+1)-2] = -1.0
            E[row, (i+1)*(kbezier+1)-1] = 1.0
            E[row, (i+1)*(kbezier+1)] = 1.0
            E[row, (i+1)*(kbezier+1)+1] = -1.0
    if continuinty_constraits_per_segment>=3:
        for i in range(int(continuity_constraints/continuinty_constraits_per_segment)):
            row = i + 2*int(continuity_constraints/continuinty_constraits_per_segment)
            E[row, (i+1)*(kbezier+1)-3] = 1.0
            E[row, (i+1)*(kbezier+1)-2] = -2.0
            E[row, (i+1)*(kbezier+1)-1] = 1.0

            E[row, (i+1)*(kbezier+1)] = -1.0
            E[row, (i+1)*(kbezier+1)+1] = 2.0
            E[row, (i+1)*(kbezier+1)+2] = -1.0
    E[continuity_constraints,0] = 1.0
    d[continuity_constraints] = points[0]
    lhs = torch.zeros([Q.shape[0] + E.shape[0], Q.shape[0] + E.shape[0]], dtype=Q.dtype, device=Q.device)
    lhs[:Q.shape[0],:Q.shape[0]] = Q
    lhs[Q.shape[0]:,:E.shape[1]] = E
    lhs[:E.shape[1], Q.shape[0]:] = E.t()
    lhs_inv = pinv(lhs)
    lhs_inv[torch.abs(lhs_inv)<1E-12]=0.0
    rhs = torch.cat([torch.matmul(HugeM.t(), points), d], dim=0)
    coefs_and_lagrange = torch.matmul(lhs_inv, rhs)
    lagrange = coefs_and_lagrange[d.shape[0]:]
    coefs = coefs_and_lagrange[:-d.shape[0]]
    return coefs, lagrange, tswitchingpoints

    
def compositeBezierEval(xstart : torch.Tensor, dx : torch.Tensor, control_points : torch.Tensor, x_eval : torch.Tensor, idxbuckets : typing.Union[torch.Tensor,None] = None) -> typing.Tuple[torch.Tensor, torch.Tensor]:

    numpoints : int = x_eval.shape[-1]
    numsplinesegments : int = control_points.shape[-3]
    kbezier : int = control_points.shape[-2] - 1
    d : int = control_points.shape[-1]

    xstart_onebatchdim : torch.Tensor = xstart.view(-1, numsplinesegments)
    batchsize : int = xstart_onebatchdim.shape[0]
    x_eval_onebatchdim : torch.Tensor = x_eval.view(batchsize, numpoints)
    control_points_onebatchdim : torch.Tensor = control_points.view(batchsize, numsplinesegments, kbezier+1, d)
    dx_onebatchdim : torch.Tensor = dx.view(batchsize, numsplinesegments)
    xend_onebatchdim = xstart_onebatchdim + dx_onebatchdim


    if idxbuckets is None:
        if batchsize == 1:
            idxbuckets_ : torch.Tensor = torch.bucketize(x_eval_onebatchdim[0], xend_onebatchdim[0], right=False).view(1, numpoints)
        else:
            idxbuckets_ : torch.Tensor = torch.stack([torch.bucketize(x_eval_onebatchdim[i], xend_onebatchdim[i], right=False) for i in range(batchsize)], dim=0)
    else:
        idxbuckets_ : torch.Tensor = idxbuckets.view(batchsize, numpoints)
        
    idxbuckets_exp = idxbuckets_.unsqueeze(-1).unsqueeze(-1).expand(batchsize, numpoints, kbezier+1, d)
    corresponding_curves = torch.gather(control_points_onebatchdim, 1, idxbuckets_exp)
    corresponding_xstart = torch.gather(xstart_onebatchdim, 1, idxbuckets_)
    corresponding_dx = torch.gather(dx_onebatchdim, 1, idxbuckets_)
    s_eval = (x_eval_onebatchdim - corresponding_xstart)/corresponding_dx
    s_eval_unsqueeze = s_eval.unsqueeze(-1)
    Mbezier = bezierM(s_eval_unsqueeze.view(-1, 1), kbezier).view(batchsize, numpoints, 1, kbezier+1)
    pointseval = torch.matmul(Mbezier, corresponding_curves).squeeze(-2)
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


def Mtk(k : int, n : int, t : torch.Tensor):
    return torch.pow(t,k)*torch.pow(1-t,(n-k))*comb(n, k, exact=True)

def bezierArcLength(control_points, d0=None, N=59, simpsonintervals=4 ):
    
    t=torch.stack([torch.linspace(0.0, 1.0, steps = simpsonintervals*N+1, dtype=control_points.dtype, device=control_points.device ) for i in range(control_points.shape[0])], dim=0)
    tsamp = t[:,[i*simpsonintervals for i in range(N+1)]]

    Mderiv, deriv = bezierDerivative(control_points, t=t, order=1)
    Mderivsamp = Mderiv[:,[i*simpsonintervals for i in range(N+1)],:]

    speeds = torch.norm(deriv,p=2,dim=2)
    

    #want [1.0, 4.0, 2.0, 4.0, 1.0] for simpsonintervals=4
    simpsonscale = torch.ones(speeds.shape[0], simpsonintervals+1, 1, dtype=speeds.dtype, device=speeds.device)
    simpsonscale[:,[i for i in range(1,simpsonintervals,2)]] = 4.0
    simpsonscale[:,[i for i in range(2,simpsonintervals,2)]] = 2.0
    Vmat = torch.stack([ torch.stack([speeds[i,simpsonintervals*j:simpsonintervals*(j+1)+1] for j in range(0, N)], dim=0)   for i in range(speeds.shape[0])], dim=0)

    relmoves = torch.matmul(Vmat, simpsonscale)[:,:,0]
    if d0 is None:
        d0_ = torch.zeros(speeds.shape[0], dtype=speeds.dtype, device=speeds.device).unsqueeze(1)
    else:
        d0_ = d0.unsqueeze(1)
    distances = torch.cat([d0_,d0_+torch.cumsum(relmoves,dim=1)/(3.0*simpsonintervals*N)],dim=1)
    
    speedsamp = speeds[:,[i*simpsonintervals for i in range(N+1)]]
    derivsamp = deriv[:,[i*simpsonintervals for i in range(N+1)]]
    return Mderivsamp, tsamp, derivsamp, speedsamp, distances

def compositeBezierSpline_with_boundary_conditions_(x : torch.Tensor, Y : torch.Tensor, boundary_conditions : torch.Tensor):
    k=3        
    d = Y.shape[1]
    if boundary_conditions.shape[0]!=2:
        raise ValueError("Must have four boundary conditions. boundary_conditions tensor is only %dx%d" % (boundary_conditions.shape[0], boundary_conditions.shape[1]))
    if boundary_conditions.shape[1]!=d:
        raise ValueError("Invalid shape of boundary conditions: %s for" +
                         "knots of dimension %d.  boundary_conditions must of size 4x%d, an initial"+
                          " velocity and acceleration followed by a final velocity and acceleration, each being %dD." % (str(boundary_conditions.shape), d, d, d))
    V0 : torch.Tensor = boundary_conditions[0]
    A0 : torch.Tensor = boundary_conditions[1]
    Vf : torch.Tensor = boundary_conditions[2]
    Af : torch.Tensor = boundary_conditions[3]

    numcurves = Y.shape[0]-1 
    dxvec = x[1:]-x[:-1]
    dx2vec = dxvec*dxvec
    bezier_control_points : torch.Tensor = torch.zeros((numcurves, k+1, d), dtype=Y.dtype, device=Y.device)
    bezier_control_points[0,0] = Y[0]
    bezier_control_points[0,1] = (dxvec[0]/k)*V0 + Y[0]
    bezier_control_points[0,2] = (dx2vec[0]/(k*(k-1)))*A0 + 2*bezier_control_points[0,1] - Y[0]
    bezier_control_points[0,3] = Y[1]

    for i in range(1, numcurves):
        bezier_control_points[i,0] = Y[i]
        bezier_control_points[i,1] = Y[i] + (dxvec[i]/dxvec[i-1])*(Y[i]-bezier_control_points[i-1,2])
        bezier_control_points[i,2] = (dx2vec[i]/dx2vec[i-1])*(Y[i] - 2.0*bezier_control_points[i-1,2] + bezier_control_points[i-1,1]) + 2.0*bezier_control_points[i,1] - Y[i] 
        bezier_control_points[i,3] = Y[i+1]
    return bezier_control_points


def compositeBezierSpline_periodic_(x : torch.Tensor, Y : torch.Tensor):
    k=3
    if torch.linalg.norm(Y[-1] - Y[0], ord=2)>1E-5:
        raise ValueError("Y[-1] and Y[0] must be the same value for a periodic spline")
    if x.shape[0]!=Y.shape[0]:
        raise ValueError("x and Y must be equal in size in dimension 0. x.shape[0]=%d, but Y.shape[0]=%d" % (x.shape[0], Y.shape[0])) 
    points = Y[:-1]
    numpoints = points.shape[0]
    d = points.shape[1]
    numparams = (k-1)*numpoints*d
    dxvec = x[1:]-x[:-1]
    dx2vec = dxvec*dxvec

    first_order_design_matrix : torch.Tensor = torch.zeros((int(numparams/2), numparams), dtype=x.dtype, device=x.device)
    first_order_bvec : torch.Tensor = torch.zeros_like(first_order_design_matrix[:,0])

    second_order_design_matrix : torch.Tensor = torch.zeros_like(first_order_design_matrix)
    second_order_bvec : torch.Tensor = torch.zeros_like(first_order_design_matrix[:,0])
    for i in range(0, first_order_design_matrix.shape[0], d):
        C0index = int(i/d)
        C1index = int(C0index+1)
        dx0 = dxvec[C0index%numpoints]
        dx1 = dxvec[C1index%numpoints]

        jstart_first_order = d*(2*C0index+1)
        dr0inv = (1/dx0)*torch.eye(d, dtype=x.dtype, device=x.device)
        first_order_design_matrix[i:i+d, jstart_first_order:jstart_first_order+d] = dr0inv
        dr1inv = (1/dx1)*torch.eye(d, dtype=x.dtype, device=x.device)
        jstart2_first_order = (jstart_first_order+d)%numparams
        first_order_design_matrix[i:i+d, jstart2_first_order:jstart2_first_order+d] = dr1inv

        first_order_bvec[i:i+d] = torch.matmul(dr0inv + dr1inv, points[C1index%numpoints])
        
        jstart_second_order = d*2*C0index
        jstart2_second_order = (jstart_second_order+d)%numparams
        jstart3_second_order = (jstart2_second_order+d)%numparams
        jstart4_second_order = (jstart3_second_order+d)%numparams

        dx0square = dx2vec[C0index%numpoints]
        dx0squareinv =  (1/dx0square)*torch.eye(d, dtype=x.dtype, device=x.device)
        second_order_design_matrix[i:i+d, jstart_second_order:jstart_second_order+d] = dx0squareinv
        second_order_design_matrix[i:i+d, jstart2_second_order:jstart2_second_order+d] = -2*dx0squareinv

        dx1square = dx2vec[C1index%numpoints]
        dx1squareinv =  (1/dx1square)*torch.eye(d, dtype=x.dtype, device=x.device)
        second_order_design_matrix[i:i+d, jstart3_second_order:jstart3_second_order+d] = 2*dx1squareinv
        second_order_design_matrix[i:i+d, jstart4_second_order:jstart4_second_order+d] = -dx1squareinv

        second_order_bvec[i:i+d] = torch.matmul(dx1squareinv - dx0squareinv, points[C1index%numpoints])
    A = torch.cat([first_order_design_matrix, second_order_design_matrix], dim=0)
    b = torch.cat([first_order_bvec, second_order_bvec], dim=0)

    res : torch.Tensor = torch.linalg.solve(A, b.unsqueeze(1))[:,0]

    all_curves = torch.cat([points.unsqueeze(1), res.reshape(numpoints , k-1, d), points[torch.linspace(1, numpoints, numpoints, dtype=torch.int64)%numpoints].unsqueeze(1)], dim=1)

    return all_curves

def bezierM(t : torch.Tensor, n : int) -> torch.Tensor:
    return torch.stack([Mtk(k,n,t) for k in range(n+1)],dim=2)

def evalBezier(M : torch.Tensor, control_points : torch.Tensor):
    return torch.matmul(M, control_points)

def evalBezierSinglePoint(s : torch.Tensor, control_points : torch.Tensor):
    num_control_points : int = control_points.shape[-2]
    order_int : int = num_control_points - 1
    return torch.matmul(bezierM(s.unsqueeze(-1), order_int), control_points).squeeze(1)
 
def bezierAntiDerivative(control_points : torch.Tensor, p0 : torch.Tensor) -> torch.Tensor:
    shapeout : list = list(control_points.shape)
    shapeout[-2]+=1
    numpoints_in = control_points.shape[-2]
    d = control_points.shape[-1]
    control_points_flat = control_points.view(-1, numpoints_in, d)
    p0flat = p0.view(-1, d)
    batchflat = p0flat.shape[0]
    cumsum_control_points = torch.cumsum(control_points_flat, 1) + p0flat.view(batchflat, 1, d).expand(batchflat, numpoints_in, d)
    antideriv_control_points = torch.cat([p0flat.view(batchflat,1,d), cumsum_control_points], dim=1)
    return antideriv_control_points.view(shapeout)

    
    
def bezierDerivative(control_points : torch.Tensor, t = None, M = None, order = 1, covariance : Union[None, torch.Tensor] = None )\
     -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    if (bool(t is not None) ^ bool(M is not None)):
        b = control_points.shape[0]
        n = control_points.shape[1]-1
        if t is not None:
            Mderiv = bezierM(t,n-order)
        else:
            Mderiv = M
        pdiff =  control_points[:,1:] - control_points[:,:-1]
        if covariance is not None:
            covardiff = covariance[:,1:] + covariance[:,:-1]
        for i in range(1,order):
            pdiff =  pdiff[:,1:] - pdiff[:,:-1]
            if covariance is not None:
                covardiff = covardiff[:,1:] + covardiff[:,:-1]
        factor = torch.prod(torch.linspace(n,n-order+1,order))
        deriv_values = factor*torch.matmul(Mderiv, pdiff)
        if covariance is None:
            return Mderiv, deriv_values
        else:
            covar_deriv = torch.square(factor)*torch.sum(Mderiv[:,:,:,None,None]*covardiff, dim=2)
            return Mderiv, deriv_values, covar_deriv    
        
        # if order%2==0:
        #     pascalrow : torch.Tensor = torch.as_tensor([np.power(-1.0, i)*nChoosek(order,i) for i in range(order+1)], dtype=control_points.dtype, device=control_points.device)
        # else:
        #     pascalrow : torch.Tensor = torch.as_tensor([np.power(-1.0, i+1)*nChoosek(order,i) for i in range(order+1)], dtype=control_points.dtype, device=control_points.device)
        # pascalmatrix : torch.Tensor = torch.zeros([b, control_points.shape[1] - order, control_points.shape[1] ], dtype=pascalrow.dtype, device=pascalrow.device)
        # for i in range(0, pascalmatrix.shape[1]):
        #     pascalmatrix[:,i,i:i+pascalrow.shape[0]] = pascalrow
        # pdiff : torch.Tensor = torch.matmul(pascalmatrix, control_points)
        # factor = torch.prod(torch.linspace(n,n-order+1,order))
        # deriv_values = factor*torch.matmul(Mderiv, pdiff)
        # d = control_points.shape[2]
        # numpoints = Mderiv.shape[1]
        # pascalmatrix_square : torch.Tensor = torch.square(pascalmatrix)
        # covariance_flat : torch.Tensor = covariance.view(b,n+1,-1)
        # pdiff_covar_flat : torch.Tensor = torch.matmul(pascalmatrix_square, covariance_flat)
        # msquare = torch.square(Mderiv)
        # covarout = torch.square(factor)*torch.matmul(msquare, pdiff_covar_flat).view(b,numpoints,d,d)
        # return Mderiv, deriv_values, covarout
    else:
        raise ValueError("One of t or M must be set, but not both")

def bezierLsqfit(points, n, t = None, M = None, built_in_lstq=False, minimum_singular_value=0.0, fix_first_point = False) -> Tuple[torch.Tensor, torch.Tensor]:
    if ((t is None) and (M is None)) or ((t is not None) and (M is not None)):
        raise ValueError("One of t or M must be set, but not both")
    if M is None:
        M_ = bezierM(t,n)
    else:
        M_ = M
    batch = M_.shape[0]
    if built_in_lstq:
        if fix_first_point:
            res = torch.cat([torch.zeros_like(points[:,0,:]).unsqueeze(1), torch.linalg.lstsq(M_[:,:,1:], points).solution], dim=1)
        else:
            res = torch.linalg.lstsq(M_, points).solution
        return M_, res
    else:
        if fix_first_point:
            res = torch.cat([torch.zeros_like(points[:,0,:]).unsqueeze(1), torch.matmul(pinv(M_[:,:,1:], minimum_singular_value=minimum_singular_value), points)], dim=1)
        else:
            res = torch.matmul(pinv(M_, minimum_singular_value=minimum_singular_value), points)
        return M_, res

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
    




class BezierCurveModule(torch.nn.Module):
    def __init__(self, control_points, mask = None):
        super(BezierCurveModule, self).__init__()
        if mask is None:
            self.mask = [True for asdf in range(control_points.shape[0])]
        else:
            self.mask = mask
        self.control_points : torch.nn.ParameterList = torch.nn.ParameterList([ torch.nn.Parameter(control_points[i], requires_grad=self.mask[i]) for i in range(len(self.mask)) ])
    @staticmethod
    def lsqFit(s, pts, n, mask=None):
        assert(s.shape[0]==pts.shape[0])
        M, cntrlpoints = bezierLsqfit(pts, n, t=s)
        return M, BezierCurveModule(cntrlpoints[0], mask=mask)
    def allControlPoints(self):
        return torch.stack([p for p in self.control_points], dim=0).unsqueeze(0)
    def forward(self, M):
        points = self.allControlPoints()
        return torch.matmul(M, points)