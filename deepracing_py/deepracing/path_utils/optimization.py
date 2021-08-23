from matplotlib.pyplot import sci
from numpy.ma.core import asarray
import scipy, numpy as np
import scipy.interpolate
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
        self.buffer = np.zeros_like(self.ds)
    def eval(self, x):
        print("Calling the LinearAccelConstraint eval function", flush=True)
        self.buffer[0:-1] = (x[1:]-x[:-1])/(2.0*self.ds[:-1])
        self.buffer[-1]=(x[0]-x[-1])/(2.0*self.ds[-1])
        return self.buffer
    def jac(self, x):
        return self.linearaccelmat
    def asSciPy(self, lb, ub, keep_feasible=False):
        return NonlinearConstraint(self.eval, lb, ub, jac = self.jac, keep_feasible=keep_feasible)

class CentripetalAccelerationContrainst():
    def __init__(self, radii: np.ndarray):
        self.invradii = 1.0/radii
        self.idx = np.arange(0, radii.shape[0], dtype=np.int64, step=1)
        speeds = np.asarray([0.0, 45.0, 130.0, 170.0, 225.0])/2.23693629
        maxcas = np.asarray([1.75, 1.75, 2.75, 4.25, 4.25])*9.81
        self.caspline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(speeds, maxcas, k=1)
        self.casplineder : scipy.interpolate.BSpline = self.caspline.derivative()
        # slope, intercept = np.polyfit(speeds, maxcas, 1)
        # self.slope = slope
        # self.intercept = intercept
    def eval(self, x):
        print("Calling the DiagonalConstraint eval function", flush=True)
        speeds = np.sqrt(x)
        # return x*self.invradii - (self.intercept + self.slope*speeds)
        return x*self.invradii - self.caspline(speeds)
    def jac(self, x):
        speeds = np.sqrt(x)
        # return scipy.sparse.dia_matrix( ( np.asarray([self.invradii - (self.slope/(2.0*speeds))]) , np.array([0], dtype=np.int64) ), shape=(x.shape[0], x.shape[0]))
        slopes = self.casplineder(speeds)
        return scipy.sparse.dia_matrix( ( np.asarray([self.invradii - (slopes/(2.0*speeds))]) , np.array([0], dtype=np.int64) ), shape=(x.shape[0], x.shape[0]))
    def asSciPy(self, keep_feasible=False):
        return NonlinearConstraint(self.eval, -1.0E10*np.ones_like(self.invradii), np.zeros_like(self.invradii) + 1.0E-5, jac = self.jac, keep_feasible=keep_feasible)

class OptimWrapper():
    def __init__(self, maxspeed : float, maxlinearaccel : float, maxbraking : float, ds : float, radii : np.ndarray, dtype=np.float32):
        self.radii = radii.astype(dtype)
        self.maxspeed = maxspeed
        self.maxlinearaccel = maxlinearaccel
        self.maxbraking = maxbraking
        if type(ds)==float:
            self.ds = ds*np.ones_like(self.radii, dtype=dtype)
        else:
            self.ds = ds.astype(dtype)
        self.grad = -np.ones_like(radii, dtype=dtype)
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
        print("Calling dat functional with counter %d. Current min speed: %f. Current max speed: %f. It has been %f seconds since the last functional call" %(self.iter_counter, np.min(np.sqrt(xcurr)), np.max(np.sqrt(xcurr)), tock-self.tick), flush=True)
        self.tick = tock
        self.iter_counter+=1
        return (-np.sum(xcurr), self.grad)

    def optimize(self, x0 = None , method="SLSQP", maxiter=20, disp=False, keep_feasible=False):
        lb = 0.001
        ub = self.maxspeed**2
        deltab = self.maxspeed-1.0
        if x0 is None:
            x0 = ((lb+0.175*deltab)**2)*np.ones_like(self.radii, dtype=self.radii.dtype)
        centripetal_accel_constraint : CentripetalAccelerationContrainst = CentripetalAccelerationContrainst(self.radii)
        linear_accel_constraint : LinearAccelConstraint = LinearAccelConstraint(self.ds)
        constraints=[]
        constraints.append(linear_accel_constraint.asSciPy(-1.0*self.maxbraking*np.ones_like(self.radii), self.maxlinearaccel*np.ones_like(self.radii), keep_feasible=keep_feasible))
        constraints.append(centripetal_accel_constraint.asSciPy(keep_feasible=keep_feasible))
        if method in ["Newton-CG", "trust-ncg", "trust-krylov", "trust-constr"]:
            hessp = self.hessp
        else:
            hessp = None
        self.tick = time.time()
        return x0, minimize(self.functional, x0, method=method, jac=True, hessp=hessp, constraints=constraints, options = {"maxiter": maxiter, "disp": disp}, bounds=Bounds(lb, ub, keep_feasible=keep_feasible))



