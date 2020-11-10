import scipy, numpy as np
from scipy.optimize import LinearConstraint, minimize, Bounds

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
        self.linearaccelmat[-1,0] = 1.0
        self.linearaccelmat*=(1.0/(2.0*ds))
        self.acentripetalmat = np.diag(1.0/radii)
    # def getLinearVelConstraint(self, keep_feasible=True):
    #     lb = np.zeros_like(self.radii)
    #     ub = (self.maxspeed**2)*np.ones_like(self.radii)
    #     return LinearConstraint(np.eye(lb.shape[0]), lb, ub, keep_feasible=keep_feasible)
    def getLinearAccelConstraint(self, keep_feasible=False):
        ub = self.maxlinearaccel*np.ones_like(self.radii)
        lb = (-ub).copy()
        return LinearConstraint(self.linearaccelmat, lb, ub, keep_feasible=keep_feasible)
    def getCentripetalAccelConstraint(self, keep_feasible=False):
        lb = np.zeros_like(self.radii)
        ub = self.maxcentripetalaccel*np.ones_like(self.radii)
        return LinearConstraint(self.acentripetalmat, lb, ub, keep_feasible=keep_feasible)
    def jac(self, xcurr):
        return -np.ones_like(xcurr)
    def functional(self, xcurr):
        return -np.sum(xcurr)
    def optimize(self, x0 = None , method="SLSQP", maxiter=5):
        lb = np.zeros_like(self.radii)
        ub = (self.maxspeed**2)*np.ones_like(self.radii)
        if x0 is None:
            x0 = 0.5*ub.copy()
        constraints = (self.getLinearAccelConstraint(), self.getCentripetalAccelConstraint())
        return x0, minimize(self.functional, x0, method=method, jac=self.jac, constraints=constraints, options = {"maxiter": maxiter}, bounds=Bounds(lb, ub, keep_feasible=True))



