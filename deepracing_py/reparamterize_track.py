from cmath import exp
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
from tqdm import tqdm


def go(argdict : dict):
    searchdirs = str.split(os.getenv("F1_TRACK_DIRS"), os.pathsep)
    trackname : str = argdict["trackname"]
    racelinefile : str = deepracing.searchForFile(trackname+"_baseline.json", searchdirs)
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
    raceline : np.ndarray = np.column_stack([racelinex, raceliney, racelinez])

    innerboundaryx : np.ndarray = np.asarray(innerboundarydict["x"])
    innerboundaryy : np.ndarray = np.asarray(innerboundarydict["y"])
    innerboundaryz : np.ndarray = np.asarray(innerboundarydict["z"])
    innerboundaryaug : np.ndarray = np.stack([innerboundaryx, innerboundaryy, innerboundaryz, np.ones_like(innerboundaryz)], axis=0)

    outerboundaryx : np.ndarray = np.asarray(outerboundarydict["x"])
    outerboundaryy : np.ndarray = np.asarray(outerboundarydict["y"])
    outerboundaryz : np.ndarray = np.asarray(outerboundarydict["z"])
    outerboundaryaug : np.ndarray = np.stack([outerboundaryx, outerboundaryy, outerboundaryz, np.ones_like(outerboundaryz)], axis=0)

    allpoints : np.ndarray = np.concatenate([raceline, innerboundaryaug[0:3].T, outerboundaryaug[0:3].T], axis=0)
    minz : float = np.min(allpoints[:,2])-10.0
    maxz : float = np.max(allpoints[:,2])+10.0
    meany : float = np.mean(allpoints[:,1])
    raceline[:,1]=meany
    innerboundaryaug[1]=meany
    outerboundaryaug[1]=meany

    racelinespline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(racelinet, raceline, k=3)
    racelinesplineder : scipy.interpolate.BSpline = racelinespline.derivative()
    racelinespline2ndder : scipy.interpolate.BSpline = racelinesplineder.derivative()
    
    # tsamp : np.ndarray = racelinet
    tsamp : np.ndarray = np.linspace(racelinet[0], racelinet[-1]+0.0375, num=10000)
    raceline : np.ndarray = racelinespline(tsamp)
    fig2 : matplotlib.figure.Figure = plt.figure()
    plt.plot(raceline[0,0], raceline[0,2], "g*")
    plt.plot(raceline[:,0], raceline[:,2], color="blue")
    plt.plot(innerboundaryaug[0], innerboundaryaug[2], color="black", alpha=0.5)
    plt.plot(outerboundaryaug[0], outerboundaryaug[2], color="black", alpha=0.5)
    plt.ylim(maxz+10.0, minz-10.0)
    plt.show()


    racelinevelvecs : np.ndarray = racelinesplineder(tsamp)
    racelinevelvecs[:,1]=0.0
    racelinetangentvecs : np.ndarray = racelinevelvecs/(np.linalg.norm(racelinevelvecs, ord=2, axis=1)[:,np.newaxis])
    
    ref : np.ndarray = np.zeros_like(racelinetangentvecs)
    ref[:,1]=1.0

    racelinelateralvecs : np.ndarray = np.cross(ref, racelinetangentvecs)

    racelineposes : np.ndarray = np.zeros((racelinevelvecs.shape[0], 4, 4), dtype=racelinevelvecs.dtype) 
    racelineposes[:,-1,-1] = 1.0
    racelineposes[:,0:3,0]=racelinetangentvecs
    racelineposes[:,0:3,1]=racelinelateralvecs
    racelineposes[:,0:3,2]=ref
    racelineposes[:,0:3,3]=raceline

    iboffsets : np.ndarray = np.zeros_like(racelineposes[:,0,0])
    oboffsets : np.ndarray = np.zeros_like(racelineposes[:,0,0])
    

    ib_kdtree : KDTree = KDTree(innerboundaryaug[0:3].T)
    ob_kdtree : KDTree = KDTree(outerboundaryaug[0:3].T)

    for i in tqdm(range(racelineposes.shape[0])):
        try:
            currentrlpose : np.ndarray = racelineposes[i]
            tmat : np.ndarray = np.linalg.inv(currentrlpose)


            closest_ib_index : int = ib_kdtree.query(currentrlpose[0:3,3])[1]
            ib_idx_samp : np.ndarray = np.arange(closest_ib_index-300, closest_ib_index+301, step=1, dtype=np.int64)%innerboundaryaug.shape[1]
            ib_samp : np.ndarray = (np.matmul(tmat, innerboundaryaug[:,ib_idx_samp])[0:3]).T
            ib_local_s : np.ndarray = np.zeros_like(ib_samp[:,0])
            ib_local_s[1:]=np.cumsum(np.linalg.norm(ib_samp[1:] - ib_samp[:-1], ord=2, axis=1))
            ib_local_s=ib_local_s-ib_local_s[int(ib_local_s.shape[0]/2)]
            ib_local_spline : scipy.interpolate.CubicSpline = scipy.interpolate.CubicSpline(ib_local_s, ib_samp) 
            ib_roots : np.ndarray = ib_local_spline.roots(extrapolate=False, discontinuity=False)[0]
            ib_root_idx : int = np.argmin(np.abs(ib_roots))
            ib_root : np.ndarray = ib_roots[ib_root_idx]
            ib_root_point : np.ndarray = ib_local_spline(ib_root)
            iboffsets[i] = ib_root_point[1]

            closest_ob_index : int = ob_kdtree.query(currentrlpose[0:3,3])[1]
            ob_idx_samp : np.ndarray = np.arange(closest_ob_index-300, closest_ob_index+301, step=1, dtype=np.int64)%outerboundaryaug.shape[1]
            ob_samp : np.ndarray = (np.matmul(tmat, outerboundaryaug[:,ob_idx_samp])[0:3]).T
            ob_local_s : np.ndarray = np.zeros_like(ob_samp[:,0])
            ob_local_s[1:]=np.cumsum(np.linalg.norm(ob_samp[1:] - ob_samp[:-1], ord=2, axis=1))
            ob_local_s=ob_local_s-ob_local_s[int(ob_local_s.shape[0]/2)]
            ob_local_spline : scipy.interpolate.CubicSpline = scipy.interpolate.CubicSpline(ob_local_s, ob_samp) 
            ob_roots : np.ndarray = ob_local_spline.roots(extrapolate=False, discontinuity=False)[0]
            ob_root_idx : int = np.argmin(np.abs(ob_roots))
            ob_root : np.ndarray = ob_roots[ob_root_idx]
            ob_root_point : np.ndarray = ob_local_spline(ob_root)
            oboffsets[i] = ob_root_point[1]

        except Exception as e:
            print(e)
            racelinesamp : np.ndarray = racelineposes[np.arange(0, 25, step=1, dtype=np.int64), :, 3]
            racelinelocal : np.ndarray = np.matmul(racelinesamp, tmat.T)[:,0:3]
            fig : matplotlib.figure.Figure = plt.figure()
            plt.plot(racelinelocal[:,1], racelinelocal[:,0])
            plt.plot(ib_samp[:,1], ib_samp[:,0])
            plt.plot(ob_samp[:,1], ob_samp[:,0])
            plt.show()

    iboutpoints : np.ndarray = raceline + racelineposes[:,0:3,1]*(iboffsets[:,np.newaxis])
    oboutpoints : np.ndarray = raceline + racelineposes[:,0:3,1]*(oboffsets[:,np.newaxis])

    delta_outer_rl : np.ndarray = oboutpoints - racelineposes[:,0:3,3]
    delta_outer_rl = delta_outer_rl/(np.linalg.norm(delta_outer_rl, ord=2, axis=1)[:,np.newaxis])

    delta_rl_inner : np.ndarray = racelineposes[:,0:3,3] - iboutpoints
    delta_rl_inner = delta_rl_inner/(np.linalg.norm(delta_rl_inner, ord=2, axis=1)[:,np.newaxis])

    dots = np.sum(delta_outer_rl*delta_rl_inner, axis=1)
    print(dots)
    print(np.min(dots))
    print(np.max(dots))



        
    
    





    # print(oboutposes[2])
    fig2 : matplotlib.figure.Figure = plt.figure()
    plt.plot(racelinex[0], racelinez[0], "g*")
    plt.quiver(raceline[:,0], raceline[:,2], racelinetangentvecs[:,0], racelinetangentvecs[:,2], angles="xy", scale=30.0)
    plt.quiver(iboutpoints[:,0], iboutpoints[:,2], racelinetangentvecs[:,0], racelinetangentvecs[:,2], angles="xy", scale=30.0, color="red")
    plt.quiver(oboutpoints[:,0], oboutpoints[:,2], racelinetangentvecs[:,0], racelinetangentvecs[:,2], angles="xy", scale=30.0, color="green")
    plt.ylim(maxz, minz)
    fig3 : matplotlib.figure.Figure = plt.figure()
    plt.plot(tsamp, iboffsets, label="Inner Boundary Offsets")
    plt.plot(tsamp, oboffsets, label="Outer Boundary Offsets")
    plt.legend()
    plt.show()

    raceline_dir : str = os.path.dirname(racelinefile)
    with open(os.path.join(raceline_dir, trackname+"_reparameterized.npz"), "wb") as f:
        np.savez(f, racelineposes=racelineposes, iboffsets=iboffsets, oboffsets=oboffsets)





if __name__=="__main__":
    import argparse 
    parser : argparse.ArgumentParser = argparse.ArgumentParser("Reparameterize a track as lateral distance from raceline to boundaries")
    parser.add_argument("trackname", type=str)
    go(vars(parser.parse_args()))