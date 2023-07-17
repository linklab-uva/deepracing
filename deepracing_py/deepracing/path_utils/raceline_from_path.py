import numpy as np
import numpy.linalg as la
import scipy, scipy.integrate
import argparse
import os
import matplotlib.pyplot as plt
import scipy.interpolate
import deepracing.path_utils.optimization 
import typing
from path_server.smooth_path_helper import SmoothPathHelper

class Writer:
    def __init__(self, argdict : dict, rsamp : np.ndarray, points : np.ndarray, ldsamp : typing.Union[np.ndarray, None] = None):
        self.argdict : dict = argdict
        self.rsamp : np.ndarray = rsamp
        self.ldsamp : typing.Union[np.ndarray, None] = ldsamp
        self.points : np.ndarray = points
        self.subtypeout = np.float32
        self.subtypes = []
        self.subtypes.append(("x", self.subtypeout, (1,)))
        self.subtypes.append(("y", self.subtypeout, (1,)))
        self.subtypes.append(("z", self.subtypeout, (1,)))
        self.subtypes.append(("arclength", self.subtypeout, (1,)))
        self.subtypes.append(("speed", self.subtypeout, (1,)))
        if self.ldsamp is not None:
            self.subtypes.append(("lapdistance", self.subtypeout, (1,)))
        self.numpytype = np.dtype(self.subtypes)
        self.structured_array : np.ndarray = np.zeros(self.points.shape[0], dtype=self.numpytype)
        self.structured_array["x"] = self.points[:,0,None].astype(self.subtypeout)
        self.structured_array["y"] = self.points[:,1,None].astype(self.subtypeout)
        self.structured_array["z"] = self.points[:,2,None].astype(self.subtypeout)
        self.structured_array["arclength"] = self.rsamp[:,None].astype(self.subtypeout)
        if self.ldsamp is not None:
            self.structured_array["lapdistance"] = self.ldsamp[:,None].astype(self.subtypeout)
    def writeLine(self, xcurr : np.ndarray):
        fileout = self.argdict["outfile"]
        self.structured_array["speed"] = np.sqrt(xcurr[:,None]).astype(self.subtypeout)
        deepracing.path_utils.structurednumpyToPCD(self.structured_array, fileout)

def normwrapper(r : typing.Union[float, np.ndarray], spline : scipy.interpolate.BSpline):
    if type(r)==float:
        return np.linalg.norm(spline(r), ord=2) 
    elif type(r)==np.ndarray:
        return np.linalg.norm(spline(r), ord=2, axis=1)
    raise ValueError("Invalid type: " + str(type(r)))
def normwrapperlambda(spline : scipy.interpolate.BSpline):
    return lambda r : np.linalg.norm(spline(r), ord=2) if type(r)==float else np.linalg.norm(spline(r), ord=2, axis=1) 

def optimizeLine(argdict : dict):
    print("Hello!", flush=True)

    inputfile = argdict["filepath"]
    numpytype, structured_array, _, _ = deepracing.path_utils.loadPCD(inputfile, align=True)
    chop = 2
    racelinein : np.ndarray = np.squeeze(np.stack([structured_array["x"], structured_array["y"], structured_array["z"]], axis=1))[:-chop]
    racelinein_helper : SmoothPathHelper = SmoothPathHelper(racelinein, k=argdict["k"], simpson_subintervals=30)
    dsin : float = argdict["ds"]
    num_evenly_spaced_points : int = int(round(racelinein_helper.distances[-1]/dsin))
    rsamp : np.ndarray = np.linspace(0.0, racelinein_helper.distances[-1], num = num_evenly_spaced_points)[:-1]
    actual_ds = float(rsamp[1] - rsamp[0])
    lapdistance : typing.Union[np.ndarray, None] = None
    spline_lapdistance : typing.Union[np.ndarray, None] = None
    ldsamp : typing.Union[np.ndarray, None] = None
    try:
        ldchop : np.ndarray  = np.squeeze(structured_array["lapdistance"])[:-chop]
        drfinal = racelinein_helper.distances[-1] - racelinein_helper.distances[-2]
        lapdistance : np.ndarray = \
            np.concatenate([ldchop, (ldchop[-1] + drfinal).reshape(1)], axis=0)
        spline_lapdistance : scipy.interpolate.Akima1DInterpolator = \
            scipy.interpolate.Akima1DInterpolator(racelinein_helper.distances, lapdistance)
        ldsamp : np.ndarray = spline_lapdistance(rsamp)
    except Exception as e:
        pass

    minspeed : float = argdict["minv"]
    maxspeed : float = argdict["maxv"]
    dsvec : np.ndarray = actual_ds*np.ones_like(rsamp)
    points_withy : np.ndarray = racelinein_helper.spline(rsamp)
    curvaturve_vecs : np.ndarray = racelinein_helper.spline_2nd_derivative(rsamp)[:,[0,2]]
    kappas : np.ndarray = np.linalg.norm(curvaturve_vecs, ord=2, axis=1)
    clamp : bool = argdict["clamp_sf"]
    idxclamp = (rsamp<50.0) + (rsamp>(racelinein_helper.distances[-1]-10.0))
    if clamp:
        kappas[idxclamp] = 0.0
    idxcheck = (rsamp<350.0) + (rsamp>(racelinein_helper.distances[-1]-350.0))
    print("Curvatures around start-finish: %s" % (str(kappas[idxcheck]),), flush=True)
    writer : Writer = Writer(argdict, rsamp, points_withy, ldsamp = ldsamp)
    print("Building the sqp object", flush=True)
    sqp = deepracing.path_utils.optimization.OptimWrapper(minspeed, maxspeed, dsvec, kappas, callback = writer.writeLine, debug=argdict["debug"])
    print("Built the sqp object", flush=True)

    cafactor=argdict["cafactor"]
    initialguessratio = argdict["initialguessratio"]
    if (initialguessratio is None) or initialguessratio<0.0:
        maxcurvature = np.max(kappas)
        x0 : np.ndarray = (cafactor*1.7*9.8/maxcurvature)*np.ones_like(rsamp)
    else:
        x0speed : float = np.clip(initialguessratio*maxspeed, minspeed, maxspeed)
        x0 : np.ndarray = (x0speed**2)*np.ones_like(rsamp)

    #method="trust-constr"
    method=argdict["method"]
    maxiter=argdict["maxiter"]
    accelfactor=argdict["accelfactor"]
    brakefactor=argdict["brakefactor"]
    hard_constraints=argdict["hard_constraints"]
    print("Running the optimization", flush=True)
    x0, optimres = sqp.optimize(maxiter=maxiter, method=method, disp=True, keep_feasible=hard_constraints, \
                 x0=x0, accelfactor=accelfactor, brakefactor=brakefactor, cafactor=cafactor, initial_guess_ratio=argdict["initialguessratio"])
    writer.writeLine(optimres.x)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="Path to .pcd file to generate optimal line from",  type=str)
    parser.add_argument("--ds", type=float, default=2.0, help="Sample the path at points this distance apart along the path")
    parser.add_argument("--maxiter", type=float, default=20, help="Maximum iterations to run the solver")
    parser.add_argument("--k", default=3, type=int, help="Degree of spline interpolation")
    parser.add_argument("--minv", default=2.0, type=float, help="Min linear speed the car can have")
    parser.add_argument("--maxv", default=90.0, type=float, help="Max linear speed the car can have")
    parser.add_argument("--method", default="SLSQP", type=str, help="Optimization method to use")
    parser.add_argument("--outfile", default="raceline_optimized.pcd", type=str, help="What to name the output file. Default is the same name as the input file")
    parser.add_argument("--initialguessratio", default=None, type=float, help="Scale factors used to determine initial guess")
    parser.add_argument("--accelfactor", default=1.0, type=float, help="Scale the max acceleration limits by this factor")
    parser.add_argument("--brakefactor", default=1.0, type=float, help="Scale the max braking limits by this factor")
    parser.add_argument("--cafactor", default=1.0, type=float,    help="Scale the max centripetal acceleration limits by this factor")
    parser.add_argument("--pca", action="store_true",  help="Project the raceline onto a PCA of the boundaries")
    parser.add_argument("--hard-constraints", action="store_true",  help="Enforce hard constraints in the optimization")
    parser.add_argument("--clamp-sf", action="store_true",  help="Clamp the curvature values to 0 around the start-finish straight")
    parser.add_argument("--debug", action="store_true",  help="Print current state of the optimization on each iteration for debugging")
    args = parser.parse_args()
    optimizeLine(vars(args))