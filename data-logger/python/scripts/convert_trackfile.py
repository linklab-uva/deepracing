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
args = parser.parse_args()
trackfilein = str(args.trackfile)
track = np.loadtxt(trackfilein,delimiter=",",skiprows=2)
print(track)

#Eigen::Vector3d p(vec[1], vec[3], vec[2]);
#rtn.push_back(std::make_pair(vec[0], p));

r = track[:,0].copy()
rlin = np.linspace(r[0],r[-1], num = len(r))
x = track[:,1].copy()
y = track[:,3].copy()
z = track[:,2].copy()

tckX = scipy.interpolate.splrep(r, x, s=0)
tckY = scipy.interpolate.splrep(r, y, s=0)
tckZ = scipy.interpolate.splrep(r, z, s=0)


xfit = scipy.interpolate.splev(rlin, tckX, der=0, ext=2)
yfit = scipy.interpolate.splev(rlin, tckY, der=0, ext=2)
zfit = scipy.interpolate.splev(rlin, tckZ, der=0, ext=2)

xdot = scipy.interpolate.splev(r, tckX, der=1, ext=2)
ydot = scipy.interpolate.splev(r, tckY, der=1, ext=2)
zdot = scipy.interpolate.splev(r, tckZ, der=1, ext=2)


xdotdot = scipy.interpolate.splev(r, tckX, der=2, ext=2)
ydotdot = scipy.interpolate.splev(r, tckY, der=2, ext=2)
zdotdot = scipy.interpolate.splev(r, tckZ, der=2, ext=2)



X = np.vstack((r,x,y,z)).transpose()
Xdot = np.vstack((xdot,ydot,zdot)).transpose()
Xdotdot = np.vstack((xdotdot,ydotdot,zdotdot)).transpose()
Xdotnorm = Xdot.copy()
for i in range(Xdotnorm.shape[0]):
    Xdotnorm[i,:] = Xdotnorm[i,:]/la.norm(Xdotnorm[i,:])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r', marker='o', s = np.ones_like(x))
ax.quiver(x, y, z, Xdotnorm[:,0], Xdotnorm[:,1], Xdotnorm[:,2])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

csvout = trackfilein + ".out.csv"
headerstring = "distance along path, x, y, z, xdot, ydot, zdot, xdotdot, ydotdot, zdotdot"
np.savetxt(csvout, np.hstack((X,Xdot,Xdotdot)), delimiter=",",header=headerstring)