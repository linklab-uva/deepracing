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


parser = argparse.ArgumentParser()
parser.add_argument("trackfile", help="Path to trackfile to convert",  type=str)
parser.add_argument("--num_samples", default=0, type=int, help="Number of values to sample from the spline. Default (0) means no sampling and just copy the data as is")
parser.add_argument("--k", default=3, type=int, help="Degree of spline interpolation, ignored if num_samples is 0")
parser.add_argument("--negate_tangents", action="store_true", help="Flip the sign all all of the tangent vectors")
args = parser.parse_args()
argdict = vars(args)
negate_tangents = argdict["negate_tangents"]
trackfilein = argdict["trackfile"]
trackdir = os.path.dirname(trackfilein)
num_samples = argdict["num_samples"]
k = argdict["k"]
trackin = np.loadtxt(trackfilein,delimiter=",",skiprows=2)
# print(trackin)
# print(trackin.shape)
I = np.argsort(trackin[:,0])
track = trackin[I].copy()
r = track[:,0].copy()
# coords = [(0, 1), (1, 0), (1, 1), (0, 0)]
# center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
# print(sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360))



Xin = np.zeros((track.shape[0],4))
#Xin[:,0] = ((r-r[0])/r[-1]).copy()
Xin[:,0] = r
Xin[:,1] = track[:,1]
Xin[:,2] = track[:,3]
Xin[:,3] = track[:,2]

final_vector = Xin[0,1:] - Xin[-1,1:]
final_distance = np.linalg.norm(final_vector)
final_unit_vector = final_vector/final_distance
nstretch = 3
rstretch =  np.linspace(final_distance/nstretch,final_distance*(nstretch-1)/nstretch,nstretch)
final_stretch = np.row_stack([Xin[-1,1:] + rstretch[i]*final_unit_vector for i in range(rstretch.shape[0])])
final_r =  rstretch + Xin[-1,0]
Xin = np.row_stack((Xin, np.column_stack((final_r,final_stretch))))

# rnormalized = Xin[:,0] - Xin[0,0]
# rnormalized = rnormalized/rnormalized[-1]
spline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(Xin[:,0], Xin[:,1:], k = k)
xfornormals = np.column_stack([Xin[:,1], np.zeros_like(Xin[:,1]), Xin[:,3]])
rsamp = np.linspace(Xin[0,0], Xin[-1,0], num = num_samples)
splinevals=spline(rsamp)
tangentspline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(Xin[:,0], xfornormals, k = 3).derivative()
tangents=tangentspline(rsamp)
tangent_norms = np.linalg.norm(tangents,axis=1)
unit_tangents=tangents/tangent_norms[:,np.newaxis]
# #print(unit_tangents[0:20])

# unit_tangents = np.column_stack([np.zeros_like(rsamp), np.zeros_like(rsamp), np.zeros_like(rsamp)])
# for i in range(0,unit_tangents.shape[0]):
#     p0 = splinevals[(i-1)%splinevals.shape[0]]
#     p1 = splinevals[i]
#     p2 = splinevals[(i+1)%splinevals.shape[0]]
#     d1 = p1-p0
#     d1 = d1/np.linalg.norm(d1)
#     d2 = p2-p1
#     d2 = d2/np.linalg.norm(d2)
#     unit_tangents[i] = 0.5*(d1+d2)





# if negate_tangents:
#     unit_tangents = -unit_tangents

# up = np.column_stack((np.zeros_like(rsamp), np.ones_like(rsamp), np.zeros_like(rsamp)))
# normals = np.cross(up,unit_tangents)
# normals = normals/np.linalg.norm(normals, axis=1)[:,np.newaxis]
# print(normals[0:20])
# rout = Xin[0,0] + tsamp*(Xin[-1,0] - Xin[0,0])
rout = rsamp# - rsamp[0]
X = np.column_stack((rout,splinevals))



x = X[:,1]
y = X[:,2]
z = X[:,3]
x_tangent = unit_tangents[:,0]
y_tangent = unit_tangents[:,1]
z_tangent = unit_tangents[:,2]
diffs = X[1:,1:] - X[0:-1,1:]
diffnorms = np.linalg.norm(diffs,axis=1)


fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
plt.xlim(np.max(x)+10, np.min(x)-10)
plt.scatter(Xin[:,1], Xin[:,3], c='b', marker='o', s = 6.0*np.ones_like(Xin[:,1]))
plt.scatter(x, z, c='r', marker='o', s =2.0*np.ones_like(x))
#plt.quiver(x, z, x_tangent, z_tangent , angles="xy", units="width")#, scale=0.0000001)
#plt.plot(x, z, c='b')#, marker='o', s = np.ones_like(x))
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
plt.show()

# armaout = os.path.join(trackdir,os.path.splitext(trackfilein)[0] + ".arma.txt")
# matout = np.hstack((np.array([r]).transpose(),X,Xdot,Xdotdot))
# headerstring = "ARMA_MAT_TXT_FN008\n" + \
#                 str(X.shape[0]) + " " + str(X.shape[1])
# np.savetxt(armaout, X, delimiter="\t", header=headerstring, comments="")
jsonout = os.path.join(trackdir,os.path.splitext(trackfilein)[0] + ".json")
jsondict : dict = {}
jsondict["dist"] = X[:,0].tolist()
jsondict["x"] = x.tolist()
jsondict["y"] = y.tolist()
jsondict["z"] = z.tolist()
jsondict["x_tangent"] = x_tangent.tolist()
jsondict["y_tangent"] = y_tangent.tolist()
jsondict["z_tangent"] = z_tangent.tolist()
with open(jsonout,"w") as f:
    json.dump( jsondict , f , indent=1 )
print("First point: " + str(X[0,:]))
print("Last point: " + str(X[-1,:]))
print("Average diff norm: " + str(np.mean(diffnorms)))
print("Final diff norm: " + str(diffnorms[-1]))
#deepracing.arma_utils.writeArmaFile(armaout, r,X,Xdot)