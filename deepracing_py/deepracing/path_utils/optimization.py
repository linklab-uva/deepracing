from numpy.ma.core import asarray
import scipy, numpy as np
from scipy.optimize import LinearConstraint, minimize, Bounds, NonlinearConstraint

import torch
from torch import Tensor
import scipy.sparse
import time
class LinearAccelConstraint():
    def __init__(self, ds: np.ndarray):
        self.ds = ds
        numpoints = self.ds.shape[0]
        data = np.empty(2*numpoints, dtype=self.ds.dtype)
        data[0:numpoints]=-np.ones_like(self.ds)/(2.0*self.ds)
        data[numpoints:-1]=np.ones_like(self.ds[0:-1])/(2.0*self.ds[0:-1])
        data[-1]=1.0/(2.0*self.ds[-1])
        row = np.empty(2*numpoints, dtype=np.int64)
        row[0:numpoints]=np.arange(0,numpoints, step=1, dtype=np.int64)
        row[numpoints:-1]=np.arange(0,numpoints-1, step=1, dtype=np.int64)
        row[-1]=numpoints-1
        col = np.empty(2*numpoints, dtype=np.int64)
        col[0:numpoints]=np.arange(0,numpoints, step=1, dtype=np.int64)
        col[numpoints:-1]=np.arange(1,numpoints, step=1, dtype=np.int64)
        col[-1]=0
        self.linearaccelmat=scipy.sparse.bsr_matrix((data, (row,col)), shape=(numpoints, numpoints), dtype=self.ds.dtype)   
     #   print(self.linearaccelmat.toarray()[[0,1,2,-3,-2,-1]])     
        self.buffer = np.zeros_like(self.ds)
    def eval(self, x):
        print("Calling the LinearAccelConstraint eval function")
        self.buffer[0:-1] = (x[1:]-x[:-1])/(2.0*self.ds[:-1])
        self.buffer[-1]=(x[0]-x[-1])/(2.0*self.ds[-1])
        return self.buffer
    def jac(self, x):
        return self.linearaccelmat
    def asSciPy(self, lb, ub, keep_feasible=False):
        return NonlinearConstraint(self.eval, lb, ub, jac = self.jac, keep_feasible=keep_feasible)

class DiagonalConstraint():
    def __init__(self, v: np.ndarray):
        self.v = v
        idx = np.arange(0, v.shape[0], dtype=np.int64, step=1)
        self.jacMat = scipy.sparse.bsr_matrix((self.v, (idx, idx)), shape=(v.shape[0], v.shape[0]))
    def eval(self, x):
        print("Calling the DiagonalConstraint eval function")
        return self.v*x
    def jac(self, x):
        return self.jacMat
    def asSciPy(self, lb, ub, keep_feasible=False):
        return NonlinearConstraint(self.eval, lb, ub, jac = self.jac, keep_feasible=keep_feasible)

class OptimWrapper():
    def __init__(self, maxspeed : float, maxlinearaccel : float, maxcentripetalaccel : float, ds : float, radii : np.ndarray, dtype=np.float32):
        self.radii = radii.astype(dtype)
        self.maxspeed = maxspeed
        self.maxlinearaccel = maxlinearaccel
        self.maxcentripetalaccel = maxcentripetalaccel
        if type(ds)==float:
            self.ds = ds*np.ones_like(self.radii, dtype=dtype)
        else:
            self.ds = ds.astype(dtype)
        self.grad = -np.ones_like(radii, dtype=dtype)#/radii.shape[0]
        self.tick = 0
        self.iter_counter = 1

    def laNegCalc(self, xcurr):
        return np.matmul(self.linearaccelmat, -xcurr) - self.maxlinearaccel
        
    def laJac(self, xcurr):
        return self.linearaccelmat

    def laNegJac(self, xcurr):
        return -self.linearaccelmat
        
    def functional(self, xcurr):
        tock = time.time()
        print("Calling dat functional with counter %d. It has been %f seconds since the last functional call" %(self.iter_counter, tock-self.tick))
        self.tick = tock
        self.iter_counter+=1
        return (-np.sum(xcurr), self.grad)

    def optimize(self, x0 = None , method="SLSQP", maxiter=20, disp=False, keep_feasible=False):
        lb = 1.0
        ub = self.maxspeed**2
        deltab = self.maxspeed-1.0
        if x0 is None:
            x0 = ((1.0+0.2*deltab)**2)*np.ones_like(self.radii, dtype=self.radii.dtype)
        centripetal_accel_constraint : DiagonalConstraint = DiagonalConstraint(1.0/self.radii)
        linear_accel_constraint : LinearAccelConstraint = LinearAccelConstraint(self.ds)
        constraints=[]
        constraints.append(linear_accel_constraint.asSciPy(-self.maxlinearaccel*np.ones_like(self.radii), self.maxlinearaccel*np.ones_like(self.radii), keep_feasible=keep_feasible))
        constraints.append(centripetal_accel_constraint.asSciPy(-50000*np.ones_like(self.radii), self.maxcentripetalaccel*np.ones_like(self.radii), keep_feasible=keep_feasible))
        if method in ["Newton-CG", "trust-ncg", "trust-krylov", "trust-constr"]:
            hessp = self.hessp
        else:
            hessp = None
        self.tick = time.time()
        return x0, minimize(self.functional, x0, method=method, jac=True, hessp=hessp, constraints=constraints, options = {"maxiter": maxiter, "disp": disp}, bounds=Bounds(lb, ub, keep_feasible=True))



