import numpy as np
import numpy.linalg as la
import quaternion
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

parser = argparse.ArgumentParser()
parser.add_argument("trackfile", help="Path to trackfile to convert",  type=str)
parser.add_argument("--add_interpolated_values", help="Path to trackfile to convert",  action="store_true")
args = parser.parse_args()
trackfilein = str(args.trackfile)
track = np.loadtxt(trackfilein,delimiter=",",skiprows=2)
print(track)

#Eigen::Vector3d p(vec[1], vec[3], vec[2]);
#rtn.push_back(std::make_pair(vec[0], p));

r = track[:,0].copy()
x = track[:,1].copy()
y = track[:,3].copy()
z = track[:,2].copy()

tckX = scipy.interpolate.splrep(r, x, s=0)
tckY = scipy.interpolate.splrep(r, y, s=0)
tckZ = scipy.interpolate.splrep(r, z, s=0)

xdot = scipy.interpolate.splev(r, tckX, der=1, ext=2)
ydot = scipy.interpolate.splev(r, tckY, der=1, ext=2)
zdot = scipy.interpolate.splev(r, tckZ, der=1, ext=2)


xdotdot = scipy.interpolate.splev(r, tckX, der=2, ext=2)
ydotdot = scipy.interpolate.splev(r, tckY, der=2, ext=2)
zdotdot = scipy.interpolate.splev(r, tckZ, der=2, ext=2)

X = np.vstack((x,y,z)).transpose()
Xdot = np.vstack((xdot,ydot,zdot)).transpose()
Xdotdot = np.vstack((xdotdot,ydotdot,zdotdot)).transpose()
if args.add_interpolated_values:
    print("Adding interpolated values")
    print("Old X shape: ", X.shape)
    print("Old Xdot shape: ", Xdot.shape)
    print("Old Xdotdot shape: ", Xdotdot.shape)
    rlin = np.linspace(r[0],r[-1], num = int(round(float(len(r))/2.0)))
    r = np.hstack((r,rlin))
    xfit = scipy.interpolate.splev(rlin, tckX, der=0, ext=2)
    yfit = scipy.interpolate.splev(rlin, tckY, der=0, ext=2)
    zfit = scipy.interpolate.splev(rlin, tckZ, der=0, ext=2)
    Xfit = np.vstack((xfit,yfit,zfit)).transpose()
    X = np.vstack((X,Xfit))
    

    xdotfit = scipy.interpolate.splev(rlin, tckX, der=1, ext=2)
    ydotfit = scipy.interpolate.splev(rlin, tckY, der=1, ext=2)
    zdotfit = scipy.interpolate.splev(rlin, tckZ, der=1, ext=2)
    Xdotfit = np.vstack((xdotfit,ydotfit,zdotfit)).transpose()
    Xdot = np.vstack((Xdot,Xdotfit))
    

    xdotdotfit = scipy.interpolate.splev(rlin, tckX, der=2, ext=2)
    ydotdotfit = scipy.interpolate.splev(rlin, tckY, der=2, ext=2)
    zdotdotfit = scipy.interpolate.splev(rlin, tckZ, der=2, ext=2)
    Xdotdotfit = np.vstack((xdotdotfit,ydotdotfit,zdotdotfit)).transpose()
    Xdotdot = np.vstack((Xdotdot,Xdotdotfit))

    I = np.argsort(r)
    r = r[I]
    X = X[I]
    Xdot = Xdot[I]
    Xdotdot = Xdotdot[I]



Xdotnorm = Xdot.copy()
for i in range(Xdotnorm.shape[0]):
    Xdotnorm[i,:] = Xdotnorm[i,:]/la.norm(Xdotnorm[i,:])
print("New X shape: ", X.shape)
print("New Xdot shape: ", Xdot.shape)
print("New Xdotdot shape: ", Xdotdot.shape)
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
matout = np.hstack((np.array([r]).transpose(),X,Xdot,Xdotdot))
headerstring = "ARMA_MAT_TXT_FN008\n" + \
                str(matout.shape[0]) + " " + str(matout.shape[1])
np.savetxt(armaout, matout, delimiter="\t", header=headerstring, comments="")