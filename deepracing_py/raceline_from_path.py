import numpy as np
import numpy.linalg as la
import scipy, scipy.integrate
import argparse
import os
import matplotlib.pyplot as plt
import scipy.interpolate
import deepracing.path_utils.optimization 
class Writer:
    def __init__(self, argdict : dict, points : np.ndarray):
        self.argdict : dict = argdict
        self.points : np.ndarray = points
    def writeLine(self, xcurr : np.ndarray):
        fileout = self.argdict["outfile"]
        deepracing.path_utils.numpyToPCD(np.sqrt(xcurr), self.points, fileout, x_name="velocity")

def normwrapper(r : float, spline : scipy.interpolate.BSpline):
    return np.linalg.norm(spline(r), ord=2, axis=1)
def normwrapperlambda(spline : scipy.interpolate.BSpline):
    return lambda r : np.linalg.norm(spline(r), ord=2, axis=1)

def go(argdict : dict):

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
    scaled_input = lapdistance_aug# / lapdistance_aug[-1]
    spline_in : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(scaled_input, racelinepath_aug[:,[0,2]], k=k, bc_type="periodic")
    
    arclengths : np.ndarray = np.zeros_like(scaled_input)
    integrand = normwrapperlambda(spline_in.derivative())
    for i in range(1, arclengths.shape[0]):
        xsamp = np.linspace(scaled_input[i-1], scaled_input[i], num=15)
        ysamp = integrand(xsamp)
        arclengths[i] = arclengths[i-1] + scipy.integrate.simpson(ysamp, x=xsamp)

    
    spline_arclength : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(arclengths, racelinepath_aug[:,[0,2]], k=k, bc_type="periodic")
    spline_tangent : scipy.interpolate.BSpline = spline_arclength.derivative()
    spline_curvature : scipy.interpolate.BSpline = spline_tangent.derivative()

    print(spline_tangent(arclengths[0]))
    print(spline_tangent(arclengths[-1]))
    print(spline_tangent(arclengths[-2]))
    
    dsin : float = argdict["ds"]
    rsamp : np.ndarray = np.arange(0.0, arclengths[-1], step = dsin)
    print(rsamp[-1], arclengths[-1])
    curvaturve_vecs : np.ndarray = spline_curvature(rsamp)
    kappas : np.ndarray = np.linalg.norm(curvaturve_vecs, ord=2, axis=1)
    iclamp : int = int(round(200/dsin))
    kappas[-iclamp:] = kappas[:iclamp] = 0.0
    radii = 1.0/kappas

    istraight : float = int(round(1000/dsin))
    # print(np.concatenate([kappas[-100:], kappas[:100]]))
    print(np.concatenate([radii[-istraight:], radii[:istraight]]))
    # print(spline_tangent(arclengths[-istraight:]))
    print(np.min(np.concatenate([radii[-istraight:], radii[:istraight]])))
    print(np.min(radii))
    maxspeed : float = argdict["maxv"]
    dsvec : np.ndarray = dsin*np.ones_like(rsamp)
    dsvec[-1] = np.linalg.norm(spline_arclength(rsamp[-1]) - spline_arclength(rsamp[0]), ord=2)
    points : np.ndarray = spline_arclength(rsamp)
    writer : Writer = Writer(argdict, points)
    sqp = deepracing.path_utils.optimization.OptimWrapper(maxspeed, dsvec, kappas, callback = writer.writeLine)

    x0 : np.ndarray = ((argdict["initialguessratio"]*maxspeed)**2)*np.ones_like(rsamp)

    #method="trust-constr"
    method=argdict["method"]
    maxiter=argdict["maxiter"]
    accelfactor=argdict["accelfactor"]
    brakefactor=argdict["brakefactor"]
    cafactor=argdict["cafactor"]
    x0, optimres = sqp.optimize(maxiter=maxiter, method=method, disp=True, keep_feasible=False, \
                 x0=x0, accelfactor=accelfactor, brakefactor=brakefactor, cafactor=cafactor, initial_guess_ratio=argdict["initialguessratio"])
    writer.writeLine(optimres.x)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="Path to .pcd file to generate optimal line from",  type=str)
    parser.add_argument("ds", type=float, help="Sample the path at points this distance apart along the path")
    parser.add_argument("--maxiter", type=float, default=20, help="Maximum iterations to run the solver")
    parser.add_argument("--k", default=3, type=int, help="Degree of spline interpolation, ignored if num_samples is 0")
    parser.add_argument("--maxv", default=86.0, type=float, help="Max linear speed the car can have")
    parser.add_argument("--method", default="SLSQP", type=str, help="Optimization method to use")
    parser.add_argument("--outfile", default="raceline_optimized.pcd", type=str, help="What to name the output file. Default is the same name as the input file")
    parser.add_argument("--initialguessratio", default=0.98, type=float, help="Scale factors used to determine initial guess")
    parser.add_argument("--accelfactor", default=1.0, type=float, help="Scale the max acceleration limits by this factor")
    parser.add_argument("--brakefactor", default=1.0, type=float, help="Scale the max braking limits by this factor")
    parser.add_argument("--cafactor", default=1.0, type=float,    help="Scale the max centripetal acceleration limits by this factor")
    parser.add_argument("--pca", action="store_true",  help="Project the raceline onto a PCA of the boundaries")
    args = parser.parse_args()
    go(vars(args))