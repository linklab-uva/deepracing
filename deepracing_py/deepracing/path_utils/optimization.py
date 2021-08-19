import scipy, numpy as np
from scipy.optimize import LinearConstraint, minimize, Bounds, NonlinearConstraint

import torch
from torch import Tensor
import scipy.sparse


class LinearConstraintTorch():
    def __init__(self, A : Tensor, keep_feasible=False, device = torch.device("cuda:0")):
        self.keep_feasible = keep_feasible
        self.device = device
        self.A = A.to(self.device)
        self.Anp = A.to(torch.device("cpu")).numpy().copy()
        self.hessmat = np.zeros_like(self.Anp)
    def eval(self, x):
        with torch.no_grad():
            rtn = torch.matmul(self.A, torch.from_numpy(x).type_as(self.A).to(self.device)).to(torch.device("cpu")).numpy()
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
        self.radii_inv = 1.0/radii
        self.radii_inv_mat = np.diag(self.radii_inv)
        self.maxspeed = maxspeed
        self.maxlinearaccel = maxlinearaccel
        self.maxcentripetalaccel = maxcentripetalaccel
        if type(ds)==float:
            self.ds = ds*np.ones_like(self.radii)
        else:
            self.ds = ds
        numpoints = radii.shape[0]
        self.linearaccelmat = np.eye(numpoints, dtype=np.float32)
        np.fill_diagonal(self.linearaccelmat,-1.0)
        np.fill_diagonal(self.linearaccelmat[:,1:],1.0)
        self.linearaccelmat[-1,0] = 1.0
        self.linearaccelmat*=(1.0/(2.0*self.ds))[:,np.newaxis]
        # self.linearaccelmat = scipy.sparse.bsr_matrix(self.linearaccelmat)

        idx = np.arange(0, radii.shape[0], dtype=np.int64, step=1)
        # self.acentripetalmat = scipy.sparse.bsr_matrix((1.0/radii, (idx, idx)), shape=(radii.shape[0], radii.shape[0]))
        self.acentripetalmat = np.diag(1.0/radii)

        self.grad = -np.ones_like(radii, dtype=np.float32)/radii.shape[0]
        self.hess = np.zeros_like(radii, dtype=np.float32)
        
    def getPostitiveLinearAccelConstraint(self, keep_feasible=False):
        return LinearConstraint(self.linearaccelmat, -self.maxlinearaccel, self.maxlinearaccel, keep_feasible=keep_feasible)
        
    def getNegativeLinearAccelConstraint(self, keep_feasible=False):
        return LinearConstraint(self.linearaccelmat, -self.maxlinearaccel, self.maxlinearaccel, keep_feasible=keep_feasible)

    def getCentripetalAccelConstraint(self, keep_feasible=False):
        return LinearConstraint(self.acentripetalmat, -1E-8, self.maxcentripetalaccel, keep_feasible=keep_feasible)

    def hessp(self, xcurr, p):
        return self.hess

    def caCalc(self, xcurr):
        return xcurr/self.radii - self.maxcentripetalaccel

    def caJac(self, xcurr):
        return self.radii_inv_mat

    def laCalc(self, xcurr):
        return np.matmul(self.linearaccelmat, xcurr) - self.maxlinearaccel

    def laNegCalc(self, xcurr):
        return np.matmul(self.linearaccelmat, -xcurr) - self.maxlinearaccel
        
    def laJac(self, xcurr):
        return self.linearaccelmat

    def laNegJac(self, xcurr):
        return -self.linearaccelmat

    def jac(self, xcurr):
        return self.grad
        
    def functional(self, xcurr):
        return (-np.mean(xcurr), self.grad)

    def optimize(self, x0 = None , method="SLSQP", maxiter=20, disp=False, keep_feasible=False):
        lb = 1.0
        ub = self.maxspeed**2
        if x0 is None:
            x0 = (((1.0+self.maxspeed)/8.0)**2)*np.ones_like(self.radii, dtype=np.float32)
        #linear_accel_constraint_torch = LinearConstraintTorch(torch.from_numpy(self.linearaccelmat.astype(np.float32)), keep_feasible=False)
        #centripetal_accel_constraint_torch = LinearConstraintTorch(torch.from_numpy(self.acentripetalmat.astype(np.float32)), keep_feasible=False)
        # constraints = (linear_accel_constraint_torch.asSciPy(-self.maxlinearaccel, self.maxlinearaccel), centripetal_accel_constraint_torch.asSciPy(0, self.maxcentripetalaccel))
        #constraints = [linear_accel_constraint_torch.asSciPy(-self.maxlinearaccel*np.ones_like(x0), self.maxlinearaccel*np.ones_like(x0)), centripetal_accel_constraint_torch.asSciPy(np.zeros_like(x0), self.maxcentripetalaccel*np.ones_like(x0))]
        constraints = [self.getCentripetalAccelConstraint(keep_feasible=keep_feasible), self.getPostitiveLinearAccelConstraint(keep_feasible=keep_feasible)]
        if method in ["Newton-CG", "trust-ncg", "trust-krylov", "trust-constr"]:
            hessp = self.hessp
        else:
            hessp = None
        return x0, minimize(self.functional, x0, method=method, jac=True, hessp=hessp, constraints=constraints, options = {"maxiter": maxiter, "disp": disp}, bounds=Bounds(lb, ub, keep_feasible=True))



