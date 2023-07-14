import numpy as np
import numpy.linalg as la
import argparse
import os
import matplotlib.pyplot as plt
from deepracing.path_utils.raceline_from_path import optimizeLine
import multiprocessing, multiprocessing.pool
import glob
def errorcb(exception):
    print(exception, flush=True)
def dummyfunc(arg):
    print(arg)
def optimizeLines(argdict : dict):
    files = glob.glob(argdict["filepaths"])
    optimdict : dict = dict()
    optimdict["ds"] = argdict["ds"]
    optimdict["maxiter"] = argdict["maxiter"]
    optimdict["k"] = argdict["k"]
    optimdict["minv"] = argdict["minv"]
    optimdict["maxv"] = argdict["maxv"]
    optimdict["method"] = argdict["method"]
    optimdict["initialguessratio"] = argdict["initialguessratio"]
    optimdict["pca"] = argdict["pca"]
    optimdict["hard_constraints"] = argdict["hard_constraints"]
    optimdict["debug"] = argdict["debug"]
    if argdict["threads"] is None:
        threads = 5*len(files)
    else:
        threads = argdict["threads"]
    with multiprocessing.pool.Pool(processes=threads) as pool:
        # mapresult : multiprocessing.pool.MapResult = pool.map_async(dummyfunc, files)
        # mapresult.wait()
        dicts : list[dict] = []
        for file in files:
            current_dict : dict = dict(optimdict)
            current_dict["filepath"] = file
            
            filebase, ext = os.path.splitext(os.path.basename(file))

            current_dict["outfile"] = os.path.join(argdict["output_dir"], filebase+"_optimized.pcd")
            current_dict["accelfactor"] = 1.0
            current_dict["brakefactor"] = 1.0
            current_dict["cafactor"] = 1.0
            dicts.append(dict(current_dict))

            current_dict["outfile"] = os.path.join(argdict["output_dir"], filebase+"_optimized_safe.pcd")
            current_dict["accelfactor"] = 0.9
            current_dict["brakefactor"] = 0.9
            current_dict["cafactor"] = 0.9
            dicts.append(dict(current_dict))

            current_dict["outfile"] = os.path.join(argdict["output_dir"], filebase+"_optimized_supersafe.pcd")
            current_dict["accelfactor"] = 0.8
            current_dict["brakefactor"] = 0.8
            current_dict["cafactor"] = 0.8
            dicts.append(dict(current_dict))

            current_dict["outfile"] = os.path.join(argdict["output_dir"], filebase+"_optimized_hypersafe.pcd")
            current_dict["accelfactor"] = 0.7
            current_dict["brakefactor"] = 0.7
            current_dict["cafactor"] = 0.7
            dicts.append(dict(current_dict))

            current_dict["outfile"] = os.path.join(argdict["output_dir"], filebase+"_optimized_ultrasafe.pcd")
            current_dict["accelfactor"] = 0.6
            current_dict["brakefactor"] = 0.6
            current_dict["cafactor"] = 0.6
            dicts.append(dict(current_dict))


        r = pool.map_async(optimizeLine, dicts, error_callback=errorcb)
        pool.close()
        pool.join()
            
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepaths", help="Glob expression for files to load", type=str)
    parser.add_argument("--output-dir", type=str, required=True, help="Where to put the optimized files")
    parser.add_argument("--threads", type=int, default=None, help="How many threads to use. Default is number of processor cores")
    parser.add_argument("--ds", type=float, default=2.5, help="Sample the path at points this distance apart along the path")
    parser.add_argument("--maxiter", type=float, default=20, help="Maximum iterations to run the solver")
    parser.add_argument("--k", default=3, type=int, help="Degree of spline interpolation, ignored if num_samples is 0")
    parser.add_argument("--minv", default=2.0, type=float, help="Min linear speed the car can have")
    parser.add_argument("--maxv", default=90.0, type=float, help="Max linear speed the car can have")
    parser.add_argument("--method", default="SLSQP", type=str, help="Optimization method to use")
    parser.add_argument("--initialguessratio", default=0.98, type=float, help="Scale factors used to determine initial guess")
    parser.add_argument("--pca", action="store_true",  help="Project the raceline onto a PCA of the boundaries")
    parser.add_argument("--hard-constraints", action="store_true",  help="Enforce hard constraints in the optimization")
    parser.add_argument("--debug", action="store_true",  help="Print current state of the optimization on each iteration for debugging")
    args = parser.parse_args()
    optimizeLines(vars(args))