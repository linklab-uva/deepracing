import numpy as np
import numpy.linalg as la
import scipy, scipy.integrate
import argparse
import os
import matplotlib.pyplot as plt
import scipy.interpolate
import deepracing.path_utils.optimization 
class Writer:
    def __init__(self, argdict : dict):
        self.argdict : dict = argdict
    def writeLine(xcurr : np.ndarray):
        pass

def go(argdict : dict):

    writer : Writer = Writer(argdict)
    # sqp = deepracing.path_utils.optimization.OptimWrapper(maxspeed, dsvec, kappas, callback = writer.writeLine)

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
    parser.add_argument("--outfile", default=None, type=str, help="What to name the output file. Default is the same name as the input file")
    parser.add_argument("--initialguessratio", default=0.98, type=float, help="Scale factors used to determine initial guess")
    parser.add_argument("--accelfactor", default=1.0, type=float, help="Scale the max acceleration limits by this factor")
    parser.add_argument("--brakefactor", default=1.0, type=float, help="Scale the max braking limits by this factor")
    parser.add_argument("--cafactor", default=1.0, type=float,    help="Scale the max centripetal acceleration limits by this factor")
    parser.add_argument("--pca", action="store_true",  help="Project the raceline onto a PCA of the boundaries")
    args = parser.parse_args()
    go(vars(args))