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
import deepracing
import Vector3dStamped_pb2
import scipy.interpolate
import deepracing.protobuf_utils
import deepracing.pose_utils as pose_utils
import time
import scipy.spatial.kdtree
from distutils.version import LooseVersion
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
def sortKey(packet):
    return packet.udp_packet.m_header.m_sessionTime
def plotRaceline(t,X,Xdot, figin=None, axin = None, label=None, c='r',marker='o'):
    Xdotnorm = Xdot.copy()
    for i in range(Xdotnorm.shape[0]):
        Xdotnorm[i,1]=0.0
        Xdotnorm[i,:] = Xdotnorm[i,:]/la.norm(Xdotnorm[i,:])
    if figin is None:
        fig = plt.figure()
    else:
        fig = figin
    if axin is None:
        ax = fig.add_subplot()
    else:
        ax = axin
    ax.scatter(X[:,0], X[:,2], c=c, marker=marker, s = np.ones_like(X[:,0]), label=label)
    #ax.quiver(X[:,0], X[:,2], Xdotnorm[:,0], Xdotnorm[:,2])
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    return fig, ax
parser = argparse.ArgumentParser()
parser.add_argument("reference", help="Path to referece trackfile",  type=str)
parser.add_argument("undertest", help="Path to trackfile under comparison to the reference",  type=str)
parser.add_argument("velocity_trace", help="Path to a file containing the setpoint and actual velocity traces",  type=str)

args = parser.parse_args()
referencefile = args.reference
undertestfile = args.undertest
velocity_trace = args.velocity_trace

tref, Xref, Xdotref = deepracing.loadArmaFile(referencefile)
tcomp, Xcomp, Xdotcomp = deepracing.loadArmaFile(undertestfile)




figref, axref = plotRaceline(tref, Xref, Xdotref, label="Reference Raceline")
figcomp, axcomp = plotRaceline(tcomp, Xcomp, Xdotcomp, figin=figref, axin = axref, c='g',marker='o', label="Pure Pursuit Raceline")
axcomp.legend()

kdtree = scipy.spatial.kdtree.KDTree(Xref)

distances, indices = kdtree.query(Xcomp)
closest_vels = Xdotref[indices]
dv = Xdotcomp - closest_vels
print(dv.shape)

speed_diffs = la.norm(dv,axis=1)
meandist = np.mean(distances)
stddist = np.std(distances)
mean_speederr = np.mean(speed_diffs)
std_speederr= np.std(speed_diffs)
print("Mean KDtree distance: %f"%(meandist))
print("Standard deviation of KDtree distances: %f"%(stddist))
print("Velocity error mean: %f"%(mean_speederr))
print("Velocity error standard deviation: %f"%(std_speederr))
dmax = meandist+6*stddist
dsmax = mean_speederr+3*std_speederr
N = 1000
kdexplot = np.linspace(0,dmax,N).reshape(-1, 1)
kdevelxplot = np.linspace(0,dsmax,N).reshape(-1, 1)

figkde, axkde = plt.subplots()
figkde.subplots_adjust(hspace=0.05, wspace=0.05)
kernel='gaussian'
kde = KernelDensity(kernel=kernel, bandwidth=0.25).fit(distances.reshape(-1, 1))
log_dens = kde.score_samples(kdexplot)
pdf = np.exp(log_dens)
axkde.plot(kdexplot[:,0], pdf, '-', label="kernel = '{0}'".format(kernel))
axkde.set_xlabel("Minimum distance to reference raceline")
axkde.set_ylabel("Probability Density")


figkdevel, axkdevel = plt.subplots()
figkdevel.subplots_adjust(hspace=0.05, wspace=0.05)
kernel='gaussian'
kdevel = KernelDensity(kernel=kernel, bandwidth=0.25).fit(speed_diffs.reshape(-1, 1))
log_densvel = kdevel.score_samples(kdevelxplot)
pdfvel = np.exp(log_densvel)
axkdevel.plot(kdevelxplot[:,0], pdfvel, '-', label="kernel = '{0}'".format(kernel))
axkdevel.set_xlabel("Difference in speed from closest point in reference raceline")
axkdevel.set_ylabel("Probability Density")



figveltrace, axveltrace = plt.subplots()
veltrace = np.loadtxt(velocity_trace,delimiter=",")
tveltrace = np.linspace(0,veltrace.shape[0]-1,veltrace.shape[0])/60.0
axveltrace.plot(tveltrace,veltrace[:,0], '-', label="setpoint", color="r")
axveltrace.plot(tveltrace, veltrace[:,1], '-', label="actual", color="g")
axveltrace.legend()
axveltrace.set_xlabel("Time")
axveltrace.set_ylabel("Speed (m/s)")





plt.show(block=True)

