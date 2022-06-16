import numpy as np
import numpy.linalg as la
import scipy, scipy.integrate
from PIL import Image as PILImage
import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate
import json
from deepracing.path_utils.optimization import OptimWrapper
import deepracing.path_utils.geometric as geometric
from shapely.geometry.polygon import Polygon
from shapely.geometry import LinearRing
from typing import List
import sklearn.decomposition
import deepracing
from functools import partial
from tqdm import tqdm

rin : np.ndarray = None 
xin : np.ndarray = None
yin : np.ndarray = None
zin : np.ndarray = None
rsamp : np.ndarray = None
radii : np.ndarray = None
dsvec : np.ndarray = None
spline : scipy.interpolate.BSpline = None
jsonout : str = None
def sensibleKnots(t, degree):
    numsamples = t.shape[0]
    knots = [ t[int(numsamples/4)], t[int(numsamples/2)], t[int(3*numsamples/4)] ]
    knots = np.r_[(t[0],)*(degree+1),  knots,  (t[-1],)*(degree+1)]
    return knots
def writeRacelineToFile(argdict : dict, velsquares : np.ndarray):
    global rin, xin, yin, zin, rsamp, spline, jsonout, radii, dsvec
    vels = np.sqrt(velsquares)
    # velinv = 1.0/vels
    # invspline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(rsamp, velinv, k=2)
    # invsplinead : scipy.interpolate.BSpline = invspline.antiderivative()
    # tparameterized = invsplinead(rsamp)
    # tparameterized = tparameterized - tparameterized[0]

    positionsradii = spline(rsamp)
    tparameterized : np.ndarray = np.zeros_like(vels)
    for i in range(1, tparameterized.shape[0]):
        tparameterized[i] = (rsamp[i] - rsamp[i-1])/vels[i-1] + tparameterized[i-1]

    finaldelta = np.linalg.norm(positionsradii[0] - positionsradii[-1], ord=2)
    positionsclosed = np.concatenate([positionsradii, positionsradii[0].reshape(1,-1)], axis=0)
    tclosed = np.concatenate([tparameterized, np.asarray([ tparameterized[-1] + finaldelta/vels[-1] ]) ])

    truespline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(tclosed, positionsclosed, bc_type="periodic")
    truesplinevel : scipy.interpolate.BSpline = truespline.derivative(nu=1)
    truesplineaccel : scipy.interpolate.BSpline = truespline.derivative(nu=2)

    nout = 4000
    tsampcheck = np.linspace(tparameterized[0], tparameterized[-1], num=radii.shape[0])
    tsamp = np.linspace(tparameterized[0], tparameterized[-1], num=nout)
    dsamp = np.linspace(rsamp[0], rsamp[-1], num=nout)

    splinevels = truesplinevel(tsampcheck)
    splinecentripetaccels = np.sum(np.square(splinevels), axis=1)/radii
    splinelinearaccels = truesplineaccel(tsampcheck)
    print("Expected lap time: %f" % (tclosed[-1],), flush=True)
    print("Lap Distance: %f" % (dsamp[-1] - dsamp[0] + dsvec[-1],), flush=True)
    print("max linear acceleration: %f" % (np.max(np.abs(splinelinearaccels))), flush=True)
    print("max centripetal acceleration: %f" % (np.max(np.abs(splinecentripetaccels))), flush=True)


    # psamp = truespline(tsamp)
    # xtrue = psamp[:,0]
    # ztrue = psamp[:,2]
    # final_stretch_samp = psamp[0] - psamp[-1]
    # print("Final position distance: %f" % (np.linalg.norm(final_stretch_samp, ord=2),), flush=True)


    # fig2 = plt.figure()
    # plt.xlim(np.max(xtrue)+10, np.min(xtrue)-10)
    # plt.plot(positionsradii[:,0],positionsradii[:,2],'r')
    # plt.scatter(xtrue[1:], ztrue[1:], c='b', marker='o', s = 16.0*np.ones_like(xtrue[1:]))
    # plt.plot(xtrue[0], ztrue[0], 'g*')
    # plt.show()


    jsondict : dict = {}
    jsondict["radii"] = radii.tolist()
    jsondict["speeds"] = vels.tolist()
    jsondict["r"] = rsamp.tolist()
    jsondict["t"] = tparameterized.tolist()
    jsondict["x"] = positionsradii[:,0].tolist()
    jsondict["y"] = positionsradii[:,1].tolist()
    jsondict["z"] = positionsradii[:,2].tolist()
    jsondict["rin"] = rin.tolist()
    jsondict["xin"] = xin.tolist()
    jsondict["yin"] = yin.tolist()
    jsondict["zin"] = zin.tolist()
    jsondict.update({key : argdict[key] for key in ["maxv", "method", "k", "ds"]})
    assert(len(jsondict["radii"]) == len(jsondict["r"]) == len(jsondict["t"]) == len(jsondict["x"]) == len(jsondict["y"]) == len(jsondict["z"]))


    with open(jsonout,"w") as f:
        json.dump( jsondict , f , indent=1 )
        


def go(argdict):
    global rin, xin, yin, zin, rsamp, spline, jsonout, radii, dsvec
    # if argdict["negate_normals"]:
    #     normalsign = -1.0
    # else:
    #     normalsign = 1.0
    trackfilein = os.path.abspath(argdict["trackfile"])
    trackname = str.split(os.path.splitext(os.path.basename(trackfilein))[0],"_")[0]


    isinnerboundary = "innerlimit" in os.path.basename(trackfilein)
    isouterboundary = "outerlimit" in os.path.basename(trackfilein)
    isracingline = not (isinnerboundary or isouterboundary)

    trackdir = os.path.abspath(os.path.dirname(trackfilein))
    print("trackdir: %s" %(trackdir,))
    ds = argdict["ds"]
    k = argdict["k"]
    if isracingline:
        searchdirs : List[str] = str.split(os.getenv("F1_TRACK_DIRS"), os.pathsep)
        print(searchdirs)
        with open(deepracing.searchForFile(trackname+"_innerlimit.json", searchdirs), "r") as f:
            d = json.load(f)
        innerboundary = np.column_stack([d[k] for k in ["x","y","z"]])
        with open(deepracing.searchForFile(trackname+"_outerlimit.json", searchdirs), "r") as f:
            d = json.load(f)
        outerboundary = np.column_stack([d[k] for k in ["x","y","z"]])
        del d
    trackfileext = str.lower(os.path.splitext(trackfilein)[1])
    if trackfileext==".csv":
        trackin = np.loadtxt(trackfilein,delimiter=";")
        trackin = trackin[:-1]
        Xin = np.zeros((trackin.shape[0],4))
        Xin[:,0] = trackin[:,0]
        Xin[:,1] = trackin[:,1]
        Xin[:,2] = 0.5*(np.mean(innerboundary[:,1]) + np.mean(outerboundary[:,1]))#np.zeros_like(trackin[:,1])
        Xin[:,3] = trackin[:,2]
        x0 = None
        minimumcurvatureguess=True
    elif trackfileext==".json":
        with open(trackfilein, "r") as f:
            d = json.load(f)
        Xin = np.column_stack([ np.asarray(d[k], dtype=np.float64) for k in ["rin", "xin", "yin", "zin"]])
        x0 = None
    else:
        trackin = np.loadtxt(trackfilein,delimiter=",",skiprows=2)
        I = np.argsort(trackin[:,0])
        track = trackin[I].copy()
        r = track[:,0].copy()
        Xin = np.zeros((track.shape[0]+1,4))
        Xin[:-1,1] = track[:,1]
        # Xin[:,2] = np.mean(track[:,3])
        Xin[:-1,2] = track[:,3]
        Xin[:-1,3] = track[:,2]
        Xin[:-1,0] = r - r[0]
        x0 = None
        Xin[-1,0] = Xin[-2,0] + np.linalg.norm(Xin[-2,1:] - Xin[0,1:], ord=2)
        Xin[-1,1:] = Xin[0,1:]

        minimumcurvatureguess=False
    if isracingline and argdict["pca"]:
        allpoints = np.concatenate([innerboundary, outerboundary], axis=0)
        print("Doing PCA projection", flush=True)
        pca : sklearn.decomposition.PCA = sklearn.decomposition.PCA(n_components=2)
        pca.fit(allpoints)
        print(pca.components_)
        normalvec : np.ndarray = np.cross(pca.components_[0], pca.components_[1])
        print(pca.explained_variance_ratio_)
        print(np.sum(pca.explained_variance_ratio_))
        Xin[:,1:] = pca.inverse_transform(pca.transform(Xin[:,1:]))
        innerboundary = pca.inverse_transform(pca.transform(innerboundary))
        outerboundary = pca.inverse_transform(pca.transform(outerboundary))
    else:
        pca = None
        normalvec : np.ndarray = np.ones(3, dtype=Xin.dtype)
        Amat : np.ndarray = np.column_stack([Xin[:,1], Xin[:,3], np.ones_like(Xin[:,1])])
        planecoefs : np.ndarray = np.matmul(np.linalg.pinv(Amat), -Xin[:,2])
        normalvec[0]=planecoefs[0]
        normalvec[2]=planecoefs[1]
        normalvec=normalvec/np.linalg.norm(normalvec, ord=2)
        if normalvec[1]<0:
            normalvec*=-1.0
    print(normalvec)
    fig1 = plt.figure()
    zmin : float = np.min(Xin[:,3])
    zmax : float = np.max(Xin[:,3])
    plt.plot(Xin[0,1], Xin[0,3], 'g*')
    plt.plot(Xin[:,1], Xin[:,3], c="blue")
    plt.ylim(zmax+10.0, zmin-10.0)
    if isracingline:
        plt.plot(innerboundary[:,0], innerboundary[:,2], c="black")
        plt.plot(outerboundary[:,0], outerboundary[:,2], c="black")
    plt.title("Input Trackfile")

    bc_type=None
    rin = Xin[:,0].copy()
    rin = rin-rin[0]
    ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if isinnerboundary:
        ref*=-1.0
    spline, Xsamp, speeds, unit_tangents, unit_normals, rsamp = geometric.computeTangentsAndNormals(rin, Xin[:,1:], k=k, ref=ref, ds=ds)
    print("Final delta: %f", (np.linalg.norm(Xsamp[0]-Xsamp[-1], ord=2),))
    print(rsamp)
    
    normaltangentdots = np.sum(unit_tangents*unit_normals, axis=1)
    if not np.all(np.abs(normaltangentdots)<=1E-6):
        raise ValueError("Something went wrong. one of the tangents is not normal to it's corresponding normal.")
    print("Max dot between normals and tangents: %f" % (np.max(normaltangentdots),), flush=True )


    offset_points = Xsamp + unit_normals*1.0


    xin = Xin[:,1]
    yin = Xin[:,2]
    zin = Xin[:,3]

    xsamp : np.ndarray = Xsamp[:,0]
    ysamp : np.ndarray = Xsamp[:,1]
    zsamp : np.ndarray = Xsamp[:,2]

    diffs = Xsamp[1:] - Xsamp[:-1]
    diffnorms = np.linalg.norm(diffs,axis=1)

    fig2 = plt.figure()
    plt.ylim(np.max(zsamp)+10, np.min(zsamp)-10)
    # ax = fig.gca(projection='3d')
    # ax.scatter(x, y, z, c='r', marker='o', s =2.0*np.ones_like(x))
    # ax.quiver(x, y, z, unit_normals[:,0], unit_normals[:,1], unit_normals[:,2], length=50.0, normalize=True)
    plt.scatter(xin, zin, c='b', marker='o', s = 16.0*np.ones_like(Xin[:,1]))
    # plt.scatter(x, z, c='r', marker='o', s = 4.0*np.ones_like(x))
    plt.plot(xsamp, zsamp, 'r')
    plt.plot(xsamp, zsamp, 'r*')
    plt.plot(xsamp[0], zsamp[0], 'g*')
    if not isracingline:
        plt.quiver(xsamp, zsamp, unit_normals[:,0], unit_normals[:,2], angles="xy", scale=4.0, scale_units="inches")
    else:
        plt.plot(innerboundary[:,0], innerboundary[:,2], c="black")
        plt.plot(outerboundary[:,0], outerboundary[:,2], c="black")
    try:
        plt.show()
    except:
        plt.close()

    print("Output shape: %s" %(str(Xsamp.shape),), flush=True)

    fileout = argdict["outfile"]
    if (fileout is not None) and os.path.isabs(fileout):
        jsonout = fileout
    else:
        resultdir = os.path.join(trackdir, "results")
        os.makedirs(resultdir, exist_ok=True)
        if (fileout is not None):
            jsonout = os.path.abspath(os.path.join(resultdir,fileout))
        else:
            jsonout = os.path.abspath(os.path.join(resultdir,os.path.splitext(os.path.basename(trackfilein))[0] + ".json"))

    print("jsonout: %s" %(jsonout,), flush=True)
    if isinnerboundary or isouterboundary:
        jsondict = dict()
        jsondict["rin"] = rin.tolist()
        jsondict["xin"] = xin.tolist()
        jsondict["yin"] = yin.tolist()
        jsondict["zin"] = zin.tolist()
        jsondict["r"] = rsamp.tolist()
        jsondict["x"] = xsamp.tolist()
        jsondict["y"] = ysamp.tolist()
        jsondict["z"] = zsamp.tolist()
        jsondict["nx"] = unit_normals[:,0].tolist()
        jsondict["ny"] = unit_normals[:,1].tolist()
        jsondict["nz"] = unit_normals[:,2].tolist()
        jsondict["k"] = k
        jsondict["ds"] = ds
        
        with open(jsonout,"w") as f:
            json.dump( jsondict , f , indent=1 )
        return


    print("Optimizing over a space of size: %d" %(rsamp.shape[0],), flush=True)
    # splineder : scipy.interpolate.BSpline = spline.derivative()
    # spline2ndder : scipy.interpolate.BSpline = splineder.derivative()

    # firstderivs : np.ndarray = splineder(rsamp)
    # firstderivnorms : np.ndarray = np.linalg.norm(firstderivs, ord=2, axis=1)
    # secondderivs : np.ndarray = spline2ndder(rsamp)
    # secondderivnorms : np.ndarray = np.linalg.norm(secondderivs, ord=2, axis=1)
    # radii = np.power(firstderivnorms,3)/np.sqrt(np.square(firstderivnorms)*np.square(secondderivnorms) - np.square(np.sum(firstderivs*secondderivs, axis=1)))

    radii = np.inf*np.ones_like(rsamp)
    searchrange : float = 15.0
    dI : int = int(round(searchrange/ds))# dI : int = 4
    for i in tqdm(range(Xsamp.shape[0]), desc="Estimating radii of curvature"):
        ilocal : np.ndarray = np.arange(i-dI, i+dI+1, step=1, dtype=np.int64)%(Xsamp.shape[0])
        Xlocal : np.ndarray = Xsamp[ilocal].T
        Xlocal = Xlocal-(Xlocal[:,dI])[:,np.newaxis]
        xvec = Xlocal[:,dI+1] - Xlocal[:,dI-1]
        xvec = xvec/np.linalg.norm(xvec, ord=2)
        yvec = np.cross(normalvec, xvec)
        yvec = yvec/np.linalg.norm(yvec, ord=2)
        zvec = np.cross(xvec,yvec)
        Xlocal = np.matmul(np.row_stack([xvec,yvec,zvec]), Xlocal)
        polynomial : np.ndarray = np.polyfit(Xlocal[0], Xlocal[1], 3)
        polynomialderiv : np.ndarray = np.polyder(polynomial)
        polynomial2ndderiv : np.ndarray = np.polyder(polynomialderiv)
        fprime : float = float(np.polyval(polynomialderiv, Xlocal[0,dI]))
        fprimeprime : float = float(np.polyval(polynomial2ndderiv, Xlocal[0,dI]))
        radii[i]=((1.0 + fprime**2)**1.5)/np.abs(fprimeprime)
    rprint = 100
    # idxhardcode = int(round(100.0/ds))
    # print("idxhardcode: %d" %(idxhardcode,), flush=True)
    # radii[0:idxhardcode] = radii[-idxhardcode:] = np.inf
    radii[radii>350.0]=np.inf


    print("First %d radii:\n%s" %(rprint, str(radii[0:rprint]),), flush=True)
    print("Final %d radii:\n%s" %(rprint, str(radii[-rprint:]),), flush=True)

    print("Min radius: %f" % (np.min(radii)), flush=True)
    print("Max radius: %f" % (np.max(radii)), flush=True)
    print("radii.shape: %s", (str(radii.shape),), flush=True)
    maxspeed = argdict["maxv"]
    dsvec = np.array((rsamp[1:] - rsamp[:-1]).tolist() + [np.linalg.norm(Xsamp[-1] - Xsamp[0])])
    #dsvec[-int(round(40/ds)):] = np.inf
    print("Final %d delta s:\n%s" %(rprint, str(dsvec[-rprint:]),))
    fig3  = plt.figure()
    plt.plot(rsamp, radii)
    plt.show()
    

    #del track, trackin, xin, yin, zin, dotsquares, Xin, rin, diffs, diffnorms, accels, accelnorms, tangents, tangentnorms, unit_tangents, unit_normals #, rsamp, Xsamp

    print("yay")
    writefunction = partial(writeRacelineToFile, argdict)
    sqp = OptimWrapper(maxspeed, dsvec, radii, callback = writefunction)


    #method="trust-constr"
    method=argdict["method"]
    maxiter=argdict["maxiter"]
    accelfactor=argdict["accelfactor"]
    brakefactor=argdict["brakefactor"]
    cafactor=argdict["cafactor"]
    x0, optimres = sqp.optimize(maxiter=maxiter, method=method, disp=True, keep_feasible=False, \
                 x0=x0, accelfactor=accelfactor, brakefactor=brakefactor, cafactor=cafactor, initial_guess_ratio=argdict["initialguessratio"])
    writefunction(optimres.x)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trackfile", help="Path to trackfile to convert",  type=str)
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