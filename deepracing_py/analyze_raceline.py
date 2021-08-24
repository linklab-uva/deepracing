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

# speeds = np.asarray(racelinedict["speeds"])
# speedsquares = np.square(speeds)
# time = np.asarray(racelinedict["t"])
# r = np.asarray(racelinedict["r"])
raceline = np.column_stack([racelinedict[k] for k in ["x","y","z"]])

allpoints = np.concatenate([outerboundary, innerboundary, raceline], axis=0)
print("Mean y: %f", np.mean(allpoints[:,1]))
print("Standard Deviation y: %f", np.std(allpoints[:,1]))

pca : sklearn.decomposition.PCA = sklearn.decomposition.PCA(n_components=2)
pca.fit(allpoints)
yvec = np.cross(-pca.components_[1], pca.components_[0])
rotmat = np.column_stack([pca.components_[0], yvec, -pca.components_[1] ])

print(pca.mean_)

allpointsroundtrip = pca.inverse_transform(pca.transform(allpoints))
I = np.argsort(allpointsroundtrip[:,1])


Aplane = torch.from_numpy(np.column_stack([allpointsroundtrip[:,0], allpointsroundtrip[:,1], np.ones_like(allpointsroundtrip[:,2])]))
if "lstsq" in dir(torch.linalg):
    solution, residuals, rank, singular_values = torch.linalg.lstsq(Aplane, -torch.from_numpy(allpointsroundtrip[:,2]))
else:
    solution, qr = torch.lstsq(-torch.from_numpy(allpointsroundtrip[:,2]), Aplane)
solution=solution.squeeze()
planeparams = torch.ones(4)
planeparams[0]=solution[0]
planeparams[1]=solution[1]
planeparams[3]=solution[2]
planeparams = torch.sign(planeparams[1])*planeparams/torch.norm(planeparams[0:3], p=2, dim=0)
print(planeparams)
meanpoint=torch.mean(torch.from_numpy(allpointsroundtrip), dim=0)
distances = torch.sum(torch.as_tensor(allpoints)*planeparams[0:3], axis=1) + planeparams[3]
print(torch.mean(distances))
print(torch.std(distances))
