from typing import Union
import numpy as np
import numpy.linalg as la
import scipy, scipy.integrate
import argparse
import os
import matplotlib.pyplot as plt
import scipy.interpolate
import deepracing.path_utils.optimization 
class Writer:
    def __init__(self, argdict : dict, rsamp : np.ndarray, ldsamp : np.ndarray, points : np.ndarray):
        self.argdict : dict = argdict
        self.rsamp : np.ndarray = rsamp
        self.ldsamp : np.ndarray = ldsamp
        self.points : np.ndarray = points
        self.subtypeout = np.float32
        self.subtypes = []
        self.subtypes.append(("x", self.subtypeout, (1,)))
        self.subtypes.append(("y", self.subtypeout, (1,)))
        self.subtypes.append(("z", self.subtypeout, (1,)))
        self.subtypes.append(("arclength", self.subtypeout, (1,)))
        self.subtypes.append(("lapdistance", self.subtypeout, (1,)))
        self.subtypes.append(("speed", self.subtypeout, (1,)))
        self.numpytype = np.dtype(self.subtypes)
        self.structured_array : np.ndarray = np.zeros(self.points.shape[0], dtype=self.numpytype)
        self.structured_array["x"] = self.points[:,0,None].astype(self.subtypeout)
        self.structured_array["y"] = self.points[:,1,None].astype(self.subtypeout)
        self.structured_array["z"] = self.points[:,2,None].astype(self.subtypeout)
        self.structured_array["lapdistance"] = self.ldsamp[:,None].astype(self.subtypeout)
        self.structured_array["arclength"] = self.rsamp[:,None].astype(self.subtypeout)
    def writeLine(self, xcurr : np.ndarray):
        fileout = self.argdict["outfile"]
        self.structured_array["speed"] = np.sqrt(xcurr[:,None]).astype(self.subtypeout)
        deepracing.path_utils.structurednumpyToPCD(self.structured_array, fileout)

def normwrapper(r : Union[float, np.ndarray], spline : scipy.interpolate.BSpline):
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
    chop = 8
    lapdistance : np.ndarray = np.squeeze(structured_array["lapdistance"])[:-chop]
    racelinepath : np.ndarray = np.squeeze(np.stack([structured_array["x"], structured_array["y"], structured_array["z"]], axis=1))[:-chop]

    if not np.all(racelinepath[0]==racelinepath[-1]):
        drfinal = np.linalg.norm(racelinepath[-1] - racelinepath[0], ord=2)
        lapdistance_aug : np.ndarray = np.concatenate([lapdistance, lapdistance[None,-1] + drfinal], axis=0)
        racelinepath_aug : np.ndarray = np.concatenate([racelinepath, racelinepath[None,0]], axis=0)
    else:
        lapdistance_aug : np.ndarray = lapdistance
        racelinepath_aug : np.ndarray = racelinepath

    k = argdict["k"]
    scaled_input = lapdistance_aug
    spline_in : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(scaled_input, racelinepath_aug[:,[0,2]], k=k, bc_type="periodic")
    
    arclengths : np.ndarray = np.zeros_like(scaled_input)
    for i in range(1, arclengths.shape[0]):
        xsamp = np.linspace(scaled_input[i-1], scaled_input[i], num=31)
        ysamp = normwrapper(xsamp, spline_in.derivative())
        arclengths[i] = arclengths[i-1] + scipy.integrate.simpson(ysamp, x=xsamp)
        # dr, _ = scipy.integrate.fixed_quad(normwrapper, scaled_input[i-1], scaled_input[i], args=(spline_in.derivative(),))
        # arclengths[i] = arclengths[i-1] + dr
    print(arclengths)
    yspline_arclength : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(arclengths, racelinepath_aug[:,1], k=k, bc_type="periodic")
    spline_arclength : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(arclengths, racelinepath_aug[:,[0,2]], k=k, bc_type="periodic")
    spline_tangent : scipy.interpolate.BSpline = spline_arclength.derivative()
    spline_curvature : scipy.interpolate.BSpline = spline_tangent.derivative()
    spline_lapdistance : scipy.interpolate.Akima1DInterpolator = scipy.interpolate.Akima1DInterpolator(arclengths, lapdistance_aug)

    
    dsin : float = argdict["ds"]
    num_evenly_spaced_points : int = int(round(arclengths[-1]/dsin))
    rsamp : np.ndarray = np.linspace(0.0, arclengths[-1], num = num_evenly_spaced_points)[:-1]
    actual_ds = float(rsamp[1] - rsamp[0])
    ldsamp : np.ndarray = spline_lapdistance(rsamp)
    # rsamp : np.ndarray = np.arange(0.0, arclengths[-1], step = dsin)
    # print(rsamp[-1], arclengths[-1])
    curvaturve_vecs : np.ndarray = spline_curvature(rsamp)
    kappas : np.ndarray = np.linalg.norm(curvaturve_vecs, ord=2, axis=1)
    idxclamp = (rsamp<200) + (rsamp>(arclengths[-1]-200))
    kappas[idxclamp] = 0.0
    minspeed : float = argdict["minv"]
    maxspeed : float = argdict["maxv"]
    dsvec : np.ndarray = actual_ds*np.ones_like(rsamp)
    points : np.ndarray = spline_arclength(rsamp)
    points_withy = np.stack([points[:,0], yspline_arclength(rsamp), points[:,1]], axis=1)
    writer : Writer = Writer(argdict, rsamp, ldsamp, points_withy)
    print("Building the sqp object", flush=True)
    sqp = deepracing.path_utils.optimization.OptimWrapper(minspeed, maxspeed, dsvec, kappas, callback = writer.writeLine, debug=argdict["debug"])
    print("Built the sqp object", flush=True)

    x0speed : float = np.clip(argdict["initialguessratio"]*maxspeed, minspeed, maxspeed)
    x0 : np.ndarray = (x0speed**2)*np.ones_like(rsamp)

    #method="trust-constr"
    method=argdict["method"]
    maxiter=argdict["maxiter"]
    accelfactor=argdict["accelfactor"]
    brakefactor=argdict["brakefactor"]
    cafactor=argdict["cafactor"]
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
    parser.add_argument("--maxv", default=86.0, type=float, help="Max linear speed the car can have")
    parser.add_argument("--method", default="SLSQP", type=str, help="Optimization method to use")
    parser.add_argument("--outfile", default="raceline_optimized.pcd", type=str, help="What to name the output file. Default is the same name as the input file")
    parser.add_argument("--initialguessratio", default=0.98, type=float, help="Scale factors used to determine initial guess")
    parser.add_argument("--accelfactor", default=1.0, type=float, help="Scale the max acceleration limits by this factor")
    parser.add_argument("--brakefactor", default=1.0, type=float, help="Scale the max braking limits by this factor")
    parser.add_argument("--cafactor", default=1.0, type=float,    help="Scale the max centripetal acceleration limits by this factor")
    parser.add_argument("--pca", action="store_true",  help="Project the raceline onto a PCA of the boundaries")
    parser.add_argument("--hard-constraints", action="store_true",  help="Enforce hard constraints in the optimization")
    parser.add_argument("--debug", action="store_true",  help="Print current state of the optimization on each iteration for debugging")
    args = parser.parse_args()
    optimizeLine(vars(args))