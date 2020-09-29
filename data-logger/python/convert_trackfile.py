import numpy as np
import numpy.linalg as la
import scipy
import skimage
import PIL
from PIL import Image as PILImage
import TimestampedPacketMotionData_pb2
import argparse
import os
import google.protobuf.json_format
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import TimestampedImage_pb2
import Pose3d_pb2
import cv2
import PoseSequenceLabel_pb2
import bisect
import FrameId_pb2
import Vector3dStamped_pb2
import scipy.interpolate
import deepracing.arma_utils
import yaml
import json
from functools import reduce
import operator
import math
import open3d
from shapely.geometry import Point as ShapelyPoint, MultiPoint#, Point2d as ShapelyPoint2d
from shapely.geometry.polygon import Polygon
from shapely.geometry import LinearRing

from scipy.spatial.transform import Rotation as Rot

parser = argparse.ArgumentParser()
parser.add_argument("trackfile", help="Path to trackfile to convert",  type=str)
parser.add_argument("num_samples", type=int, help="Number of values to sample from the spline. Default (0) means no sampling and just copy the data as is")
parser.add_argument("--k", default=3, type=int, help="Degree of spline interpolation, ignored if num_samples is 0")
#parser.add_argument("--negate_normals", action="store_true", help="Flip the sign all all of the computed normal vectors")
args = parser.parse_args()
argdict = vars(args)
# if argdict["negate_normals"]:
#     normalsign = -1.0
# else:
#     normalsign = 1.0
trackfilein = argdict["trackfile"]
isracingline = "racingline" in os.path.basename(trackfilein)
innerboundary = "innerlimit" in os.path.basename(trackfilein)
outerboundary = not (isracingline or innerboundary)


trackdir = os.path.dirname(trackfilein)
num_samples = argdict["num_samples"]
k = argdict["k"]
trackin = np.loadtxt(trackfilein,delimiter=",",skiprows=2)

I = np.argsort(trackin[:,0])
track = trackin[I].copy()
r = track[:,0].copy()


if isracingline:
    Xin = np.zeros((track.shape[0]-1,4))
    Xin[:,1] = track[:-1,1]
    Xin[:,2] = track[:-1,3]
    Xin[:,3] = track[:-1,2]
    Xin[:,0] = np.hstack( [np.zeros(1), np.cumsum(np.linalg.norm(Xin[1:,1:] - Xin[:-1,1:], axis=1, ord=2))   ]   )
else:
    Xin = np.zeros((track.shape[0],4))
    Xin[:,1] = track[:,1]
    Xin[:,2] = track[:,3]
    Xin[:,3] = track[:,2]
    Xin[:,0] = np.hstack( [np.zeros(1), np.cumsum(np.linalg.norm(Xin[1:,1:] - Xin[:-1,1:], axis=1, ord=2))   ]   )
Xin[:,0] = Xin[:,0] - Xin[0,0]
final_vector = Xin[0,1:] - Xin[-1,1:]
final_distance = np.linalg.norm(final_vector)
print("initial final distance: %f" %(final_distance,))
final_unit_vector = final_vector/final_distance
nstretch = 3
rstretch =  np.linspace(final_distance/nstretch,final_distance,nstretch)
final_stretch = np.row_stack([Xin[-1,1:] + rstretch[i]*final_unit_vector for i in range(rstretch.shape[0])])
final_r =  rstretch + Xin[-1,0]
Xin = np.row_stack((Xin, np.column_stack((final_r,final_stretch))))

# rnormalized = Xin[:,0] - Xin[0,0]
# rnormalized = rnormalized/rnormalized[-1]
spline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(Xin[:,0], Xin[:,1:], k = k)

finalidx = -1
if finalidx is None:
    finalextrassamps = 0
else:
    finalextrassamps = abs(finalidx)

rsamp = np.linspace(Xin[0,0], Xin[-1,0], num = num_samples + finalextrassamps)
splinevals=spline(rsamp)[0:finalidx]
rout = rsamp[0:finalidx]

splinevalspolygon = spline(np.linspace(Xin[0,0], Xin[-1,0], num = 750) )
# lr = LinearRing([(splinevalspolygon[i,0], splinevalspolygon[i,2]) for i in range(splinevalspolygon.shape[0])])
lr = LinearRing([(Xin[i,1], Xin[i,3]) for i in range(0,Xin.shape[0],2)])
polygon : Polygon = Polygon(lr)
assert(polygon.is_valid)


deltasamples = (np.arange(0, splinevals.shape[0], step=1, dtype=np.uint64) + 1)%splinevals.shape[0]
P2 = splinevals[deltasamples].copy()
P1 = splinevals.copy()
#P2[:,1] = P1[:,1] = 0.0
deltas = P2-P1
delta_norms = np.linalg.norm(deltas, axis=1, ord=2)
unit_tangents=deltas/delta_norms[:,np.newaxis]



#if innerboundary:
#     rotation = Rot.from_rotvec(np.array([0.0,0.0,-np.pi/2]))
# else:
#     rotation = Rot.from_rotvec(np.array([0.0,0.0,np.pi/2]))
# unit_normals = np.matmul(rotation.as_matrix(), unit_tangents.transpose()).transpose()

ref = np.column_stack([np.zeros_like(unit_tangents.shape[0]), np.ones_like(unit_tangents.shape[0]), np.zeros_like(unit_tangents.shape[0])]).astype(np.float64)
if innerboundary:
    ref[:,1]*=-1.0
v1 = np.cross(unit_tangents, ref)
v1 = v1/np.linalg.norm(v1, axis=1, ord=2)[:,np.newaxis]
v2 =  np.cross(v1, unit_tangents)
v2 = v2/np.linalg.norm(v2, axis=1, ord=2)[:,np.newaxis]
unit_normals = np.cross(v2, unit_tangents)
unit_normals = unit_normals/np.linalg.norm(unit_normals, axis=1, ord=2)[:,np.newaxis]



normaltangentdots = np.sum(unit_tangents*unit_normals, axis=1)
print(normaltangentdots)
if not np.all(np.abs(normaltangentdots)<=1E-6):
    raise ValueError("Something went wrong. one of the tangents is not normal to it's corresponding normal.")

print("Max dot between normals and tangents: %f" % (np.max(normaltangentdots),) )


# assert(unit_normals.shape[0]==splinevals.shape[0])
# offset_points = splinevals + unit_normals*0.1
# if innerboundary:
#     wrongside = np.array( [ not polygon.contains(ShapelyPoint(splinevals[i,0], splinevals[i,2]))  for i in range(splinevals.shape[0]) ] )
# else:
#     wrongside = np.array( [ polygon.contains(ShapelyPoint(splinevals[i,0], splinevals[i,2]))  for i in range(splinevals.shape[0]) ] )
# unit_normals[wrongside]*=-1.0

offset_points = splinevals + unit_normals*1.0




# normals = np.cross(up,unit_tangents)
# if "innerlimit" in os.path.basename(trackfilein):
#     normals*=-1.0


X = np.column_stack((rout,splinevals))
x = X[:,1]
y = X[:,2]
z = X[:,3]
x_tangent = unit_tangents[:,0]
y_tangent = unit_tangents[:,1]
z_tangent = unit_tangents[:,2]
x_normal = unit_normals[:,0]
y_normal = unit_normals[:,1]
z_normal = unit_normals[:,2]
diffs = X[1:,1:] - X[0:-1,1:]
diffnorms = np.linalg.norm(diffs,axis=1)


fig = plt.figure()
plt.xlim(np.max(x)+10, np.min(x)-10)
# ax = fig.gca(projection='3d')
# ax.scatter(x, y, z, c='r', marker='o', s =2.0*np.ones_like(x))
# ax.quiver(x, y, z, unit_normals[:,0], unit_normals[:,1], unit_normals[:,2], length=50.0, normalize=True)
plt.scatter(Xin[:,1], Xin[:,3], c='b', marker='o', s = 16.0*np.ones_like(Xin[:,1]))
plt.scatter(x, z, c='r', marker='o', s = 4.0*np.ones_like(x))
plt.plot(x[0], z[0], 'g*')
plt.quiver(x, z, unit_normals[:,0], unit_normals[:,2], angles="xy", scale=4.0, scale_units="inches")
plt.show()

jsonout = os.path.join(trackdir,os.path.splitext(trackfilein)[0] + ".json")
jsondict : dict = {}
jsondict["dist"] = X[:,0].tolist()
jsondict["x"] = x.tolist()
jsondict["y"] = y.tolist()
jsondict["z"] = z.tolist()
jsondict["x_tangent"] = x_tangent.tolist()
jsondict["y_tangent"] = y_tangent.tolist()
jsondict["z_tangent"] = z_tangent.tolist()
jsondict["x_normal"] = x_normal.tolist()
jsondict["y_normal"] = y_normal.tolist()
jsondict["z_normal"] = z_normal.tolist()
with open(jsonout,"w") as f:
    json.dump( jsondict , f , indent=1 )
print("First point: " + str(X[0,:]))
print("Last point: " + str(X[-1,:]))
print("Average diff norm: " + str(np.mean(diffnorms)))
print("Final diff norm: " + str(np.linalg.norm(X[0,1:] - X[-1,1:])))
