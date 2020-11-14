import scipy, numpy as np
from scipy.optimize import LinearConstraint, minimize, Bounds, NonlinearConstraint

import torch
from torch import Tensor


class LinearConstraintTorch():
    def __init__(self, A : Tensor, keep_feasible=False, device = torch.device("cuda:0")):
        self.keep_feasible = keep_feasible
        self.device = device
        self.A = A.double().to(self.device)
        self.Anp = A.to(torch.device("cpu")).numpy().copy()
        self.hessmat = np.zeros_like(self.Anp)
    def eval(self, x):
        with torch.no_grad():
            rtn = torch.matmul(self.A, torch.from_numpy(x).to(self.device)).to(torch.device("cpu")).numpy()
        return rtn
    def jac(self, x):
        return self.Anp
    def hess(self, x, v):
        return self.hessmat
    def asSciPy(self, lb, ub):
        return NonlinearConstraint(self.eval, lb, ub, jac = self.jac, keep_feasible=self.keep_feasible)

class OptimWrapper():
    def __init__(self, maxspeed : float, maxlinearaccel : float, maxcentripetalaccel : float, ds : float, radii : np.ndarray):
        self.radii = radii
        self.maxspeed = maxspeed
        self.maxlinearaccel = maxlinearaccel
        self.maxcentripetalaccel = maxcentripetalaccel
        self.ds = ds
        numpoints = radii.shape[0]
        self.linearaccelmat = np.eye(numpoints)
        np.fill_diagonal(self.linearaccelmat,-1.0)
        np.fill_diagonal(self.linearaccelmat[:,1:],1.0)
        self.linearaccelmat*=(1.0/(2.0*ds))
        self.linearaccelmat[-1:,:] = 0.0
        # self.linearaccelmat[-1,0] = 1.0
        self.acentripetalmat = np.diag(1.0/radii)
    #    self.acentripetalmat[-1,:] = 0.0
        
    def getLinearAccelConstraint(self, keep_feasible=False):
        return LinearConstraint(self.linearaccelmat, -self.maxlinearaccel, self.maxlinearaccel, keep_feasible=keep_feasible)

    def getCentripetalAccelConstraint(self, keep_feasible=False):
        return LinearConstraint(self.acentripetalmat, -1E-8, self.maxcentripetalaccel, keep_feasible=keep_feasible)

    def hessp(self, xcurr, p):
        return np.zeros_like(xcurr)

    def jac(self, xcurr):
        return -np.ones_like(xcurr)
        
    def functional(self, xcurr):
        return -np.sum(xcurr)

    def optimize(self, x0 = None , method="SLSQP", maxiter=20, disp=False):
        lb = 1.0
        ub = self.maxspeed**2
        if x0 is None:
            x0 = 0.5*ub*np.ones_like(self.radii)
        # linear_accel_constraint_torch = LinearConstraintTorch(torch.from_numpy(self.linearaccelmat), keep_feasible=False)
        # centripetal_accel_constraint_torch = LinearConstraintTorch(torch.from_numpy(self.acentripetalmat), keep_feasible=False)
        # constraints = (linear_accel_constraint_torch.asSciPy(-self.maxlinearaccel, self.maxlinearaccel), centripetal_accel_constraint_torch.asSciPy(0, self.maxcentripetalaccel))
        constraints = (self.getLinearAccelConstraint(), self.getCentripetalAccelConstraint())
        if method in ["Newton-CG", "trust-ncg", "trust-krylov", "trust-constr"]:
            hessp = self.hessp
        else:
            hessp = None
        return x0, minimize(self.functional, x0, method=method, jac=self.jac, hessp=hessp, constraints=constraints, options = {"maxiter": maxiter, "disp": disp}, bounds=Bounds(lb, ub, keep_feasible=True))



