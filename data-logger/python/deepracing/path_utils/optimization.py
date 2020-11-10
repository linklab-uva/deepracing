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
        
    def getLinearAccelConstraint(self, keep_feasible=False):
        return LinearConstraint(self.linearaccelmat, -self.maxlinearaccel, self.maxlinearaccel, keep_feasible=keep_feasible)

    def getCentripetalAccelConstraint(self, keep_feasible=False):
        return LinearConstraint(self.acentripetalmat, 0, self.maxcentripetalaccel, keep_feasible=keep_feasible)

    def jac(self, xcurr):
        return -np.ones_like(xcurr)
        
    def functional(self, xcurr):
        return -np.sum(xcurr)

    def optimize(self, x0 = None , method="SLSQP", maxiter=5):
        if x0 is None:
            x0 = 0.5*(self.maxspeed**2)*np.ones_like(self.radii)
        constraints = (self.getLinearAccelConstraint(), self.getCentripetalAccelConstraint())
        return x0, minimize(self.functional, x0, method=method, jac=self.jac, constraints=constraints, options = {"maxiter": maxiter}, bounds=Bounds(0, self.maxspeed**2, keep_feasible=True))



