from typing import Callable
from matplotlib.pyplot import sci
from numpy.core.numeric import ones_like
from numpy.ma.core import asarray
import scipy, numpy as np
import scipy.interpolate
from scipy.optimize import LinearConstraint, minimize, Bounds, NonlinearConstraint
import torch
from torch import Tensor
import scipy.sparse
import time

np.set_printoptions(linewidth=165, edgeitems=6)
def generate_linear_accel_mat(ds:  np.ndarray):
    numpoints = ds.shape[0]
    data = np.empty(2*numpoints, dtype=ds.dtype)
    data[0:numpoints]=-np.ones_like(ds)/(2.0*ds)
    data[numpoints:-1]=np.ones_like(ds[0:-1])/(2.0*ds[0:-1])
    data[-1]=1.0/(2.0*ds[-1])
    row = np.empty(2*numpoints, dtype=np.int64)
    row[0:numpoints]=np.arange(0,numpoints, step=1, dtype=np.int64)
    row[numpoints:-1]=np.arange(0,numpoints-1, step=1, dtype=np.int64)
    row[-1]=numpoints-1
    col = np.empty(2*numpoints, dtype=np.int64)
    col[0:numpoints]=np.arange(0,numpoints, step=1, dtype=np.int64)
    col[numpoints:-1]=np.arange(1,numpoints, step=1, dtype=np.int64)
    col[-1]=0
    return scipy.sparse.bsr_matrix((data, (row,col)), shape=(numpoints, numpoints), dtype=ds.dtype)

class BrakingConstraint():
    def __init__(self, ds: np.ndarray, max_speed : float, factor=1.0, debug : bool = False):
        self.ds = ds
        self.linearaccelmat=generate_linear_accel_mat(ds)
        self.buffer = np.zeros_like(self.ds)
        # speeds = np.asarray([        0.0, 25.0, 30.0, 40.0, 60.0, 84.0, 150.0])
        # braking_limits = np.asarray([7.5, 7.5,  16.5, 21.0, 34.5, 40.0, 40.0])*factor
        speeds : np.ndarray = np.flip(np.asarray([         125.0,  84.00,  46.00,  17.5,  0.00 ], dtype=ds.dtype))
        braking_limits : np.ndarray  = np.flip(-np.array([ 40.00,  39.00,  32.75,  16.75, 14.5 ],  dtype=ds.dtype)*factor)

        self.braking_spline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(speeds, braking_limits, k=1)
        self.braking_spline_der : scipy.interpolate.BSpline = self.braking_spline.derivative()
        print(self.linearaccelmat.toarray(), flush=True)
        self.debug : bool = debug
        self.iter_counter : int = 1

    def eval(self, x):
        if self.debug and (self.iter_counter==1 or ((self.iter_counter%10)==0)):
            print(flush=True)
            print("Calling the BrakingConstraint eval function", flush=True)
        accels = self.linearaccelmat*x
        speeds = np.sqrt(x)
        braking_limits = self.braking_spline(speeds)
        self.buffer = braking_limits - accels 
        imin = np.argmax(self.buffer)
        if self.debug and (self.iter_counter==1 or ((self.iter_counter%10)==0)):
            print("Max constraint value: %f" % (self.buffer[imin],), flush=True)
            print("Braking limit at min constraint value: %f" % (braking_limits[imin],), flush=True)
            print("Linear acceleration at min constraint value: %f" % (accels[imin],), flush=True)
            print("Speed at min constraint value: %f" % (speeds[imin],), flush=True)
            print(flush=True)
        self.iter_counter+=1
        return self.buffer
    def jac(self, x):
        speeds = np.sqrt(x)
        slopes = self.braking_spline_der(speeds)
        return -self.linearaccelmat + scipy.sparse.dia_matrix( ( np.asarray( [(slopes/(2.0*speeds))] ) , np.array([0], dtype=np.int64) ), shape=(x.shape[0], x.shape[0]) )
    def asSciPy(self, keep_feasible=False):
        return NonlinearConstraint(self.eval, -1E15*np.ones_like(self.ds), np.zeros_like(self.ds), jac = self.jac, keep_feasible=keep_feasible)

class LinearAccelConstraint():
    def __init__(self, ds: np.ndarray, max_speed : float, factor=1.0, debug : bool = False):
        self.ds = ds
        self.linearaccelmat=generate_linear_accel_mat(ds)
        self.buffer = np.zeros_like(self.ds)
        speeds = np.asarray([              0.00,  36.0,  91.5,  150.0])
        forward_accel_limits = np.asarray([17.5,  17.25,  0.00,  0.000])*factor
        self.forward_accel_spline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(speeds, forward_accel_limits, k=1)
        self.forward_accel_spline_der : scipy.interpolate.BSpline = self.forward_accel_spline.derivative()
        self.debug : bool = debug
        self.iter_counter : int = 1
     #   print(self.linearaccelmat.toarray()[[0,1,2,3,-4,-3,-2,-1]], flush=True)

    def eval(self, x):
        if self.debug and (self.iter_counter==1 or ((self.iter_counter%10)==0)):
            print(flush=True)
            print("Calling the LinearAccelConstraint eval function", flush=True)
        accels = self.linearaccelmat*x
        speeds = np.sqrt(x)
        max_accels = self.forward_accel_spline(speeds)
        self.buffer = accels - max_accels
        imax = np.argmax(self.buffer)
        if self.debug and (self.iter_counter==1 or ((self.iter_counter%10)==0)):
            print("Max constraint value : %f" % (self.buffer[imax],), flush=True)
            print("Linear acceleration limit at max constraint value: %f" % (max_accels[imax],), flush=True)
            print("Linear acceleration at max constraint value: %f" % (accels[imax],), flush=True)
            print("Speed at max constraint value: %f" % (speeds[imax],), flush=True)
            print(flush=True)
        self.iter_counter+=1
        return self.buffer
    def jac(self, x):
        speeds = np.sqrt(x)
        slopes = self.forward_accel_spline_der(speeds)
        return self.linearaccelmat - scipy.sparse.dia_matrix( ( np.asarray( [(slopes/(2.0*speeds))] ) , np.array([0], dtype=np.int64) ), shape=(x.shape[0], x.shape[0]))
    def asSciPy(self, keep_feasible=False):
        return NonlinearConstraint(self.eval, -1E15*np.ones_like(self.ds), np.zeros_like(self.ds), jac = self.jac, keep_feasible=keep_feasible)

class CentripetalAccelerationConstraint():
    def __init__(self, kappas : np.ndarray, maxspeed : float, factor=1.0, debug : bool = False):
        self.kappas = kappas
        self.idx = np.arange(0, kappas.shape[0], dtype=np.int64, step=1)
        maxspeedmph = 2.2369362920544025*maxspeed
        print("Max speed in MPH: %f" % (maxspeedmph,), flush=True)
        self.caspline : scipy.interpolate.BSpline = CentripetalAccelerationConstraint.limitspline(factor=factor)
        self.casplineder : scipy.interpolate.BSpline = self.caspline.derivative()
        self.debug : bool = debug
        self.iter_counter : int = 1
    @staticmethod
    def limitspline(factor = 1.0) -> scipy.interpolate.BSpline:
        speeds = np.asarray([-100.00, 0.00,  45.0,  60.0,  130.0,  170.0,  190.0,  225.0], dtype=np.float64)/2.2369362920544025 #mph to m/s
        maxcas = np.asarray([1.75,    1.75,  2.00,  2.50,  3.250,  3.250,  3.500,  3.500], dtype=np.float64)*9.81*factor #Gforce to m/s^2
        return scipy.interpolate.make_interp_spline(speeds, maxcas, k=1)
    def eval(self, x):
        if self.debug and (self.iter_counter==1 or ((self.iter_counter%10)==0)):
            print(flush=True)
            print("Calling the CentripetalAccelerationConstraint eval function", flush=True)
        speeds = np.sqrt(x)
        centripetal_accels = x*self.kappas
        limits = self.caspline(speeds)
        rtn = centripetal_accels - limits
        imax = np.argmax(rtn)
        if self.debug and (self.iter_counter==1 or ((self.iter_counter%10)==0)):
            print("Max constraint value: %f" % (rtn[imax],), flush=True)
            print("Centripetal acceleration limit at max constraint value: %f" % (limits[imax],), flush=True)
            print("Centripetal acceleration at max constraint value: %f" % (centripetal_accels[imax],), flush=True)
            print("Speed at max constraint value: %f" % (speeds[imax],), flush=True)
            print("Radius of curvature at max constraint value: %f" % (1.0/self.kappas[imax],), flush=True)
            print(flush=True)
        self.iter_counter+=1
        return rtn
    def jac(self, x):
        speeds = np.sqrt(x)
        slopes = self.casplineder(speeds)
        return scipy.sparse.dia_matrix( ( np.asarray( [self.kappas - (slopes/(2.0*speeds))] ) , np.array([0], dtype=np.int64) ), shape=(x.shape[0], x.shape[0]))
    def asSciPy(self, keep_feasible=False):
        return NonlinearConstraint(self.eval, -1E15*np.ones_like(self.kappas), np.zeros_like(self.kappas), jac = self.jac, keep_feasible=keep_feasible)
        # return NonlinearConstraint(self.eval, -5.0*9.81*np.ones_like(self.kappas), np.zeros_like(self.kappas), jac = self.jac, keep_feasible=keep_feasible)

class OptimWrapper():
    def __init__(self, minspeed : float, maxspeed : float, ds : float, radii : np.ndarray, dtype=np.float32, callback  = None, debug : bool = False):
        self.radii = radii.astype(dtype)
        self.minspeed = minspeed
        self.maxspeed = maxspeed
        if type(ds)==float:
            self.ds = ds*np.ones_like(self.radii, dtype=dtype)
        else:
            self.ds = ds.astype(dtype)
        self.grad = -np.ones_like(radii, dtype=dtype)
        self.tick = 0
        self.iter_counter = 1
        self.callback = callback
        self.debug : bool = debug
        
    def functional(self, xcurr):
        if (self.callback is not None) and self.iter_counter>2:
            self.callback(xcurr)
        tock = time.time()
        speeds = np.sqrt(xcurr)
        if self.debug and (self.iter_counter==1 or ((self.iter_counter%50)==0)):
            idxmin = -int(200.0/self.ds[0])
            idxmax = int(200.0/self.ds[0])
            idx = np.arange(idxmin,idxmax,step=1,dtype=np.int64)
            print(flush=True)
            print("Calling dat functional with counter %d. Current min speed: %f. Current max speed: %f. It has been %f seconds since the last functional call" %(self.iter_counter, np.min(speeds), np.max(speeds), tock-self.tick), flush=True)
            print("Speeds around start-finish:\n%s" % (str(speeds[idx]),))
            print(flush=True)
        self.tick = tock
        self.iter_counter+=1
        return (-np.sum(xcurr), self.grad)
    def hess(self, xcurr : np.ndarray):
        return np.zeros((xcurr.shape[0], xcurr.shape[0]), dtype=xcurr.dtype)
    def hessp(self, xcurr, p):
        return np.zeros_like(xcurr)

    def optimize(self, x0 = None , method="SLSQP", maxiter=20, disp=False, keep_feasible=False, accelfactor=1.0, brakefactor=1.0, cafactor=1.0, initial_guess_ratio=0.99):
        lb = np.square(self.minspeed*np.ones_like(self.radii, dtype=self.radii.dtype))
        ub = np.square(self.maxspeed*np.ones_like(self.radii, dtype=self.radii.dtype))
        if x0 is None:
            x0 = np.square(initial_guess_ratio*self.maxspeed*np.ones_like(self.radii, dtype=self.radii.dtype))
        centripetal_accel_constraint : CentripetalAccelerationConstraint = CentripetalAccelerationConstraint(self.radii, self.maxspeed, factor=cafactor, debug=self.debug)
        braking_constraint : BrakingConstraint = BrakingConstraint(self.ds, self.maxspeed, factor=brakefactor, debug=self.debug)
        linear_accel_constraint : LinearAccelConstraint = LinearAccelConstraint(self.ds, self.maxspeed, factor=accelfactor, debug=self.debug)
        constraints=[]
        constraints.append(linear_accel_constraint.asSciPy(keep_feasible=keep_feasible))
        constraints.append(braking_constraint.asSciPy(keep_feasible=keep_feasible))
        constraints.append(centripetal_accel_constraint.asSciPy(keep_feasible=keep_feasible))
        if method in ["Newton-CG", "trust-ncg", "trust-krylov", "trust-constr"]:
            hess = self.hess
            hessp = self.hessp
        else:
            hess = None
            hessp = None
        self.tick = time.time()
        return x0, minimize(self.functional, x0, method=method, jac=True, hess=hess, constraints=constraints, options = {"maxiter": maxiter, "disp": disp}, bounds=Bounds(lb, ub, keep_feasible=keep_feasible))



