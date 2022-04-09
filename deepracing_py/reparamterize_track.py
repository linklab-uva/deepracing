import numpy as np
import numpy.linalg as la
import scipy, scipy.interpolate
import scipy.spatial.transform 
from scipy.spatial import KDTree
import json
import deepracing
import os
import matplotlib.figure
import matplotlib.pyplot as plt

def go(argdict : dict):
    searchdirs = str.split(os.getenv("F1_TRACK_DIRS"), os.pathsep)
    trackname : str = argdict["trackname"]
    racelinefile : str = deepracing.searchForFile(trackname+"_minimumcurvaturebaseline.json", searchdirs)
    with open(racelinefile, "r") as f:
        racelinedict : dict = json.load(f)

    innerboundaryfile : str = deepracing.searchForFile(trackname+"_innerlimit.json", searchdirs)
    with open(innerboundaryfile, "r") as f:
        innerboundarydict : dict = json.load(f)

    outerboundaryfile : str = deepracing.searchForFile(trackname+"_outerlimit.json", searchdirs)
    with open(outerboundaryfile, "r") as f:
        outerboundarydict : dict = json.load(f)

    racelinet : np.ndarray = np.asarray(racelinedict["t"])
    racelinex : np.ndarray = np.asarray(racelinedict["x"])
    raceliney : np.ndarray = np.asarray(racelinedict["y"])
    racelinez : np.ndarray = np.asarray(racelinedict["z"])
    raceline : np.ndarray = np.column_stack([racelinex, racelinez])

    innerboundaryx : np.ndarray = np.asarray(innerboundarydict["x"])
    innerboundaryy : np.ndarray = np.asarray(innerboundarydict["y"])
    innerboundaryz : np.ndarray = np.asarray(innerboundarydict["z"])
    innerboundary : np.ndarray = np.column_stack([innerboundaryx, innerboundaryz])
    innerboundaryaug : np.ndarray = np.stack([innerboundary[:,0], innerboundary[:,1], np.ones_like(innerboundary[:,1])], axis=0)

    outerboundaryx : np.ndarray = np.asarray(outerboundarydict["x"])
    outerboundaryy : np.ndarray = np.asarray(outerboundarydict["y"])
    outerboundaryz : np.ndarray = np.asarray(outerboundarydict["z"])
    outerboundary : np.ndarray = np.column_stack([outerboundaryx, outerboundaryz])
    outerboundaryaug : np.ndarray = np.stack([outerboundary[:,0], outerboundary[:,1], np.ones_like(outerboundary[:,1])], axis=0)

    allpoints : np.ndarray = np.concatenate([raceline, innerboundary, outerboundary], axis=0)
    minx : float = np.min(allpoints[:,0])-10.0
    maxx : float = np.max(allpoints[:,0])+10.0

    racelinespline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(racelinet, raceline, k=3)
    racelinesplineder : scipy.interpolate.BSpline = racelinespline.derivative()
    racelinespline2ndder : scipy.interpolate.BSpline = racelinesplineder.derivative()

    racelinevelvecs : np.ndarray = racelinesplineder(racelinet)
    racelinetangentvecs : np.ndarray = racelinevelvecs/(np.linalg.norm(racelinevelvecs, ord=2, axis=1)[:,np.newaxis])
    racelinetangentvecs_full : np.ndarray = np.column_stack([racelinetangentvecs[:,0], racelinetangentvecs[:,1], np.zeros_like(racelinetangentvecs[:,0])])
    racelinelateralvecs : np.ndarray = racelinetangentvecs[:,[1,0]]
    racelinelateralvecs[:,1]*=-1.0
    # up : np.ndarray = np.zeros_like(racelinetangentvecs_full)
    # up[:,2]=1.0
    # racelinelateralvecs_full : np.ndarray = np.cross(up, racelinetangentvecs_full)
    # racelinelateralvecs : np.ndarray = racelinelateralvecs_full[:,0:2]
    racelineposes : np.ndarray = np.zeros((racelinevelvecs.shape[0], 3, 3), dtype=racelinevelvecs.dtype) 
    racelineposes[:,2,2] = 1.0
    racelineposes[:,0:2,0] = racelinetangentvecs
    racelineposes[:,0:2,1] = racelinelateralvecs
    racelineposes[:,0:2,2] = raceline
    print(racelineposes)

    ib_kdtree : KDTree = KDTree(innerboundary)
    ob_kdtree : KDTree = KDTree(outerboundary)

    for i in range(10):
        tmat : np.ndarray = np.linalg.inv(racelineposes[i])

        iblocal : np.ndarray = np.matmul(tmat, innerboundaryaug).T
        ibdistances : np.ndarray = np.linalg.norm(iblocal, ord=2, axis=1)
        closest_ib_index : int = np.argmin(ibdistances)
        ib_idx_samp : np.ndarray = np.arange(closest_ib_index-10, closest_ib_index+11, step=1, dtype=np.int64)%iblocal.shape[0]

        oblocal : np.ndarray = np.matmul(tmat, outerboundaryaug).T
        obdistances : np.ndarray = np.linalg.norm(oblocal, ord=2, axis=1)
        closest_ob_index : int = np.argmin(obdistances)
        ob_idx_samp : np.ndarray = np.arange(closest_ob_index-10, closest_ob_index+11, step=1, dtype=np.int64)%oblocal.shape[0]
        print()
        print(iblocal[ib_idx_samp][5])
        print(oblocal[ob_idx_samp][5])
        # break
    
    







    fig : matplotlib.figure.Figure = plt.figure()
    plt.plot(racelinex[0], racelinez[0], "g*")
    plt.plot(racelinex, racelinez)
    plt.plot(innerboundaryx, innerboundaryz, c="black")
    plt.plot(outerboundaryx, outerboundaryz, c="black")
    plt.xlim(maxx, minx)
    plt.show()


if __name__=="__main__":
    import argparse 
    parser : argparse.ArgumentParser = argparse.ArgumentParser("Reparameterize a track as lateral distance from raceline to boundaries")
    parser.add_argument("trackname", type=str)
    go(vars(parser.parse_args()))