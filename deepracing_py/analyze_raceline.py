import numpy.linalg as la
import numpy as np
from numpy.ma.core import asarray
from scipy.spatial.transform import Rotation as Rot
from matplotlib import pyplot as plt
import argparse
import json
import scipy.interpolate, scipy.stats
import os
import sklearn.decomposition
import torch
import deepracing
# import torch.linalg

parser = argparse.ArgumentParser(description="Look at some statistical metrics of the optimal raceline")
parser.add_argument("racelinefile", type=str, help="Path to the raceline")
args = parser.parse_args()
argdict=vars(args)

racelinefile = os.path.abspath(argdict["racelinefile"])
racelinebase = os.path.basename(racelinefile)
racelinedir = os.path.dirname(racelinefile)
with open(racelinefile, "r") as f:
    racelinedict = json.load(f)
raceline = np.column_stack([racelinedict[k] for k in ["x","y","z"]])
trackname = str.split(os.path.splitext(racelinebase)[0],"_")[0]
innerboundaryfile = deepracing.searchForFile(trackname+"_innerlimit.json", str.split(os.getenv("F1_TRACK_DIRS"), os.pathsep))
outerboundaryfile = deepracing.searchForFile(trackname+"_outerlimit.json", str.split(os.getenv("F1_TRACK_DIRS"), os.pathsep))
with open(innerboundaryfile, "r") as f:
    innerboundarydict = json.load(f)
with open(outerboundaryfile, "r") as f:
    outerboundarydict = json.load(f)
innerboundary=np.column_stack([innerboundarydict[k] for k in ["x","y","z"]])
outerboundary=np.column_stack([outerboundarydict[k] for k in ["x","y","z"]])
allpoints = np.concatenate([outerboundary, innerboundary, raceline], axis=0)
pca = sklearn.decomposition.PCA(n_components=2)
pca.fit(allpoints)
ref = np.cross(pca.components_[0], pca.components_[1])
ref = np.sign(ref[1])*ref/np.linalg.norm(ref, ord=2)
print("Reference vector: %s" %(str(ref),))

speeds = np.asarray(racelinedict["speeds"])
speedsquares = np.square(speeds)
r = np.asarray(racelinedict["r"])
ds = np.empty_like(r)
ds[:-1] = r[1:]-r[:-1]
ds[-1]=np.linalg.norm(raceline[0]-raceline[-1], ord=2)
linearaccels = np.empty_like(ds)
linearaccels[0:-1]=(speedsquares[1:]-speedsquares[:-1])/(2.0*ds[:-1])
linearaccels[-1]=(speedsquares[0]-speedsquares[-1])/(2.0*ds[-1])
print("Min Linear Acceleration (from data): %f" % (np.min(linearaccels),))
print("Max Linear Acceleration (from data): %f" % (np.max(linearaccels),))

prspline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(r, raceline)
rdotspline : scipy.interpolate.BSpline = prspline.derivative()
rdotdotspline : scipy.interpolate.BSpline = rdotspline.derivative()

rdot = rdotspline(r)
unit_tangents = rdot/np.linalg.norm(rdot,ord=2,axis=1)[:,np.newaxis]
rdotdot = rdotdotspline(r)

refdots=np.sum(np.cross(rdot, rdotdot)*ref, axis=1)
# refdots[np.abs(refdots)<1E-2]=0.0
directions = np.sign(refdots)
radii = (np.linalg.norm(rdot, ord=2, axis=1)**3)/(np.linalg.norm(np.cross(rdot, rdotdot), ord=2, axis=1))
#print(radii)
centripetalaccels = speedsquares/radii
print("Min Centripetal Acceleration (from data): %f" % (np.min(centripetalaccels),))
print("Max Centripetal Acceleration (from data): %f" % (np.max(centripetalaccels),))



normals = np.row_stack([np.cross(ref, unit_tangents[i]) for i in range(unit_tangents.shape[0])])
unit_normals = normals/np.linalg.norm(normals,ord=2,axis=1)[:,np.newaxis]


# vrspline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(r-r[0], speeds)
velinvrspline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(r-r[0], 1.0/speeds, k=1)
time = velinvrspline.antiderivative()(r-r[0])
time = time-time[0]
print("Estimated laptime: %f" % (time[-1] + ds[-1]/speeds[-1],))

raceline = np.column_stack([racelinedict[k] for k in ["x","y","z"]])
#
# print(allpoints)

pca : sklearn.decomposition.PCA = sklearn.decomposition.PCA(n_components=2)
pca.fit(allpoints)
yvec = np.cross(-pca.components_[1], pca.components_[0])
rotmat = np.column_stack([pca.components_[0], yvec, -pca.components_[1] ])

timespline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(time, raceline, k=5)
velspline : scipy.interpolate.BSpline = timespline.derivative(nu=1)
accelspline : scipy.interpolate.BSpline = velspline.derivative(nu=1)
tsamp = np.linspace(time[0], time[-1], 1000)
fig = plt.figure()
plt.plot(time, speeds, label="Speeds In")
plt.plot(tsamp, np.linalg.norm(velspline(tsamp), ord=2, axis=1), label="Spline Speeds")
plt.legend()
fig2 = plt.figure()
plt.plot(time, linearaccels, label="Linear Accels (from data)")
plt.plot(time, np.sum(accelspline(time)*unit_tangents, axis=1), label="Spline Linear Accels")
plt.legend()
acentspline = np.sum(accelspline(time)*unit_normals, axis=1)
# nomatch = np.sign(acentspline)!=directions
# print(np.sum(nomatch))
# nomatchidx = np.asarray([i for i in range(nomatch.shape[0]) if nomatch[i]])
# # print(directions[nomatchidx]*centripetalaccels[nomatchidx])
# # print(acentspline[nomatchidx])
fig3 = plt.figure()
plt.plot(time, centripetalaccels, label="Centripetal Accels (from data)")
plt.plot(time, np.abs(acentspline), label="Spline Centripetal Accels")
plt.legend()
fig4 = plt.figure()
plt.plot(raceline[:,0], raceline[:,2], label="Raceline")
plt.legend()
plt.plot(innerboundary[:,0], innerboundary[:,2], c="black")
plt.plot(outerboundary[:,0], outerboundary[:,2], c="black")
plt.plot(raceline[0,0], raceline[0,2], "g*")
# plt.scatter(raceline[nomatchidx,0], raceline[nomatchidx,2], marker="X", c="red")
plt.xlim(np.max(allpoints[:,0]) + 10.0, np.min(allpoints[:,0])-10.0)
plt.show()