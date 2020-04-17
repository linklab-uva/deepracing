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
parser = argparse.ArgumentParser()
parser.add_argument("trackfile", help="Path to trackfile to convert",  type=str)
parser.add_argument("--add_interpolated_values", help="Path to trackfile to convert",  action="store_true")
args = parser.parse_args()
trackfilein = str(args.trackfile)
trackin = np.loadtxt(trackfilein,delimiter=",",skiprows=2)
print(trackin)
print(trackin.shape)
I = np.argsort(trackin[:,0])
track = trackin[I].copy()
r = track[:,0]



X = np.zeros((track.shape[0],3))
X[:,0] = track[:,1]
X[:,1] = track[:,3]
X[:,2] = track[:,2]

x = X[:,0]
y = X[:,1]
z = X[:,2]
print(X)
print(X.shape)


spline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(r, X)

splinedot = spline.derivative(nu=1)
splinedotdot = spline.derivative(nu=2)

Xdot = splinedot(r)


xdot = Xdot[:,0]
ydot = Xdot[:,1]
zdot = Xdot[:,2]


xdotdot = Xdot[:,0]
ydotdot = Xdot[:,1]
zdotdot = Xdot[:,2]


if args.add_interpolated_values:
    print("Adding interpolated values")
    rlin = np.linspace(r[0],r[-1], num = int(round(float(len(r))/2.0)))
    Xspl = spline(rlin)
    Xdotspl = splinedot(rlin)
    Xstack = np.vstack((X,Xspl))
    Xdotstack = np.vstack((Xdot,Xdotspl))
    rstack = np.hstack((r,rlin))
    Istack = np.argsort(rstack)
    r = rstack[Istack]
    X = Xstack[Istack]
    Xdot = Xdotstack[Istack]




Xdotnorm = Xdot.copy()
for i in range(Xdotnorm.shape[0]):
    Xdotnorm[i,:] = Xdotnorm[i,:]/la.norm(Xdotnorm[i,:])
print("New X shape: ", X.shape)
print("New Xdot shape: ", Xdot.shape)
print("New Xdotdotnorm shape: ", Xdotnorm.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c='r', marker='o', s = np.ones_like(x))
ax.quiver(X[:,0], X[:,1], X[:,2], Xdotnorm[:,0], Xdotnorm[:,1], Xdotnorm[:,2])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

armaout = os.path.splitext(trackfilein)[0] + ".arma.txt"
# matout = np.hstack((np.array([r]).transpose(),X,Xdot,Xdotdot))
# headerstring = "ARMA_MAT_TXT_FN008\n" + \
#                 str(matout.shape[0]) + " " + str(matout.shape[1])
# np.savetxt(armaout, matout, delimiter="\t", header=headerstring, comments="")
deepracing.arma_utils.writeArmaFile(armaout, r,X,Xdot)