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

parser = argparse.ArgumentParser()
parser.add_argument("trackfile", help="Path to trackfile to convert",  type=str)
parser.add_argument("--num_samples", default=0, type=int, help="Number of values to sample from the spline. Default (0) means no sampling and just copy the data as is")
parser.add_argument("--k", default=3, type=int, help="Degree of spline interpolation, ignored if num_samples is 0")
args = parser.parse_args()
argdict = vars(args)
trackfilein = argdict["trackfile"]
num_samples = argdict["num_samples"]
k = argdict["k"]
trackin = np.loadtxt(trackfilein,delimiter=",",skiprows=2)
print(trackin)
print(trackin.shape)
I = np.argsort(trackin[:,0])
track = trackin[I].copy()
r = track[:,0].copy()



Xin = np.zeros((track.shape[0],4))
Xin[:,0] = ((r-r[0])/r[-1]).copy()
Xin[:,1] = track[:,1]
Xin[:,2] = track[:,3]
Xin[:,3] = track[:,2]


if num_samples>0:
    spline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(Xin[:,0], Xin[:,1:], k = k)
    tsamp = np.linspace(0, 1, num = num_samples)
    splinevals=spline(tsamp)
    rout = tsamp*r[-1]
    X = np.column_stack((rout,splinevals))
    # X[:,0] = tsamp
    # X[:,1:] = splinevals
else:
    X = Xin.copy()
    X[:,0] = r.copy()



x = X[:,1]
y = X[:,2]
z = X[:,3]
print(X)
print(X.shape)


fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
plt.scatter(Xin[:,1], Xin[:,3], c='b', marker='o', s = 2.0*np.ones_like(x))
plt.scatter(x, z, c='r', marker='o', s = 2.0*np.ones_like(x))
#plt.plot(x, z, c='b')#, marker='o', s = np.ones_like(x))
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
plt.show()

armaout = os.path.splitext(trackfilein)[0] + ".arma.txt"
#matout = np.hstack((np.array([r]).transpose(),X,Xdot,Xdotdot))
headerstring = "ARMA_MAT_TXT_FN008\n" + \
                str(X.shape[0]) + " " + str(X.shape[1])
np.savetxt(armaout, X, delimiter="\t", header=headerstring, comments="")
jsonout = os.path.splitext(trackfilein)[0] + ".json"
with open(jsonout,"w") as f:
    jsondict : dict = {}
    jsondict["dist"] = X[:,0].tolist()
    jsondict["x"] = X[:,1].tolist()
    jsondict["y"] = X[:,2].tolist()
    jsondict["z"] = X[:,3].tolist()
    json.dump( jsondict , f , indent=1 )
    
#deepracing.arma_utils.writeArmaFile(armaout, r,X,Xdot)