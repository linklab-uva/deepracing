import numpy.linalg as la
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from matplotlib import pyplot as plt
import argparse
import json
import scipy.interpolate, scipy.stats
import os
import sklearn.decomposition
import torch
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
trackname = str.split(os.path.splitext(racelinebase)[0],"_")[0]
innerboundaryfile = os.path.join(racelinedir, trackname+"_innerlimit.json")
outerboundaryfile = os.path.join(racelinedir, trackname+"_outerlimit.json")
with open(innerboundaryfile, "r") as f:
    innerboundarydict = json.load(f)
with open(outerboundaryfile, "r") as f:
    outerboundarydict = json.load(f)
innerboundary=np.column_stack([innerboundarydict[k] for k in ["x","y","z"]])
outerboundary=np.column_stack([outerboundarydict[k] for k in ["x","y","z"]])

speeds = np.asarray(racelinedict["speeds"])
speedsquares = np.square(speeds)
time = np.asarray(racelinedict["t"])
r = np.asarray(racelinedict["r"])
raceline = np.column_stack([racelinedict[k] for k in ["x","y","z"]])
#
allpoints = np.concatenate([outerboundary, innerboundary, raceline], axis=0)
# print(allpoints)

pca : sklearn.decomposition.PCA = sklearn.decomposition.PCA(n_components=2)
pca.fit(allpoints)
yvec = np.cross(-pca.components_[1], pca.components_[0])
rotmat = np.column_stack([pca.components_[0], yvec, -pca.components_[1] ])

print(pca.mean_)
# print(pca.components_)
# print(Rot.from_matrix(rotmat).as_euler("xyz"))

allpointsroundtrip = pca.inverse_transform(pca.transform(allpoints))
I = np.argsort(allpointsroundtrip[:,1])
# print(allpointsroundtrip[I])


Aplane = torch.from_numpy(np.column_stack([allpointsroundtrip[:,0], allpointsroundtrip[:,1], np.ones_like(allpointsroundtrip[:,2])]))
# solution, residuals, rank, singular_values = np.linalg.lstsq(Aplane, -allpointsroundtrip[:,2])
solution, residuals, rank, singular_values = torch.linalg.lstsq(Aplane, -torch.from_numpy(allpointsroundtrip[:,2]))
# solution, qr = torch.lstsq(torch.from_numpy(-allpointsroundtrip[:,2]),  Aplane)
solution=solution.squeeze()
# print(residuals)
planeparams = torch.ones(4)
planeparams[0]=solution[0]
planeparams[1]=solution[1]
planeparams[3]=solution[2]
planeparams = torch.sign(planeparams[1])*planeparams/torch.norm(planeparams[0:3], p=2, dim=0)
print(planeparams)
meanpoint=torch.mean(torch.from_numpy(allpointsroundtrip), dim=0)
# print(meanpoint)
# print(planeparams[0]*meanpoint[0] + planeparams[2]*meanpoint[2])
distances = torch.sum(torch.as_tensor(allpointsroundtrip)*planeparams[0:3], axis=1) + planeparams[3]
print(torch.mean(distances))
print(torch.std(distances))
# rspline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(r, raceline)
# rdotspline : scipy.interpolate.BSpline = rspline.derivative(nu=1)
# rdotdotspline : scipy.interpolate.BSpline = rspline.derivative(nu=2)
# tangents = rdotspline(r)
# unit_tangents = tangents/np.linalg.norm(tangents,ord=2,axis=1)[:,np.newaxis]

# velspline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(time, unit_tangents*speeds[:,np.newaxis]) 
# antiderspline : scipy.interpolate.BSpline = velspline.antiderivative()
# positionspline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(time, raceline) 
# positionssamp = positionspline(time)
# accelspline : scipy.interpolate.BSpline = velspline.derivative(nu=1)

# splinevels = velspline(time)
# splinespeeds = np.linalg.norm(splinevels, ord=2, axis=1)
# splinespeedsquares = np.square(splinespeeds)
# splineaccels = accelspline(time)
# refvec = np.asarray([0.0, 1.0, 0.0])

# unit_normals = np.row_stack([np.cross(refvec, unit_tangents[i]) for i in range(unit_tangents.shape[0])])
# unit_normals = unit_normals/np.linalg.norm(unit_normals, ord=2, axis=1)[:,np.newaxis]

# # splineradii = (splinespeedsquares*splinespeeds)/(np.linalg.norm(np.cross(splinevels, splineaccels), ord=2, axis=1) + 1E-6)
# # centripetalaccels = splinespeedsquares/splineradii

# linearaccels = np.sum(splineaccels*unit_tangents, axis=1)
# centripetalaccels = np.sum(splineaccels*unit_normals, axis=1)

# fig = plt.figure()
# plt.plot(time, linearaccels)
# fig2 = plt.figure()
# plt.plot(time, centripetalaccels)
# fig3 = plt.figure()
# rlintegral = raceline[0,np.newaxis] + antiderspline(time)
# plt.plot(positionssamp[:,0], positionssamp[:,2], label="Spline Fit")
# plt.plot(raceline[:,0], raceline[:,2], label="Original Raceline")
# plt.legend()
# plt.plot(innerboundary[:,0], innerboundary[:,2], c="black")
# plt.plot(outerboundary[:,0], outerboundary[:,2], c="black")
# plt.show()