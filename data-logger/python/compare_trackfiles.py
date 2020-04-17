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
import deepracing
import Vector3dStamped_pb2
import scipy.interpolate
import deepracing.protobuf_utils
import deepracing.pose_utils as pose_utils
import time
import scipy.spatial.kdtree
from scipy.spatial.kdtree import KDTree
from distutils.version import LooseVersion
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import deepracing, deepracing.arma_utils
import shapely
from shapely.geometry import Polygon, Point
import deepracing.evaluation_utils as eval_utils
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
parser.add_argument("reference", help="Path to reference trackfile",  type=str)
parser.add_argument("undertest", help="Path to trackfile under comparison to the reference",  type=str)
parser.add_argument("output_dir", help="Where to save the figures",  type=str)

args = parser.parse_args()
referencefile = args.reference
undertestfile = args.undertest
output_dir = args.output_dir

tref, Xref, Xdotref = deepracing.arma_utils.readArmaFile(referencefile)
tcomp, Xcomp, Xdotcomp = deepracing.arma_utils.readArmaFile(undertestfile)

reference_polygon : Polygon = Polygon(Xref[:,[0,2]])
distancespoly = np.nan*tcomp.copy()
distanceskdtree = np.nan*tcomp.copy()
kdtree = KDTree(Xref[:,[0,2]])
for i in range(tcomp.shape[0]):
    pointnp = Xcomp[i,[0,2]]
    pointshapely : Point = Point(pointnp)
    distancespoly[i] = eval_utils.polyDist(reference_polygon,pointshapely)
    d, min_idx = kdtree.query(pointnp)
    distanceskdtree[i] = d
mindistkdtree = np.min(distanceskdtree)
maxdistkdtree = np.max(distanceskdtree)
meandistkdtree = np.mean(distanceskdtree)
stdevkdtree= np.std(distanceskdtree)

mindistpoly = np.min(distancespoly)
maxdistpoly = np.max(distancespoly)
meandistpoly = np.mean(distancespoly)
stdevpoly = np.std(distancespoly)
print("Via KD-Tree:")
print("Min: %f"%(mindistkdtree,))
print("Max: %f"%(maxdistkdtree,))
print("Mean: %f"%(meandistkdtree,))
print("Via Polygon:")
print("Min: %f"%(mindistpoly,))
print("Max: %f"%(maxdistpoly,))
print("Mean: %f"%(meandistpoly,))

figref, axref = plotRaceline(tref, Xref, Xdotref, label="Reference Raceline")
figcomp, axcomp = plotRaceline(tcomp, Xcomp, Xdotcomp, figin=figref, axin = axref, c='g',marker='o', label="Pure Pursuit Raceline")

axref.legend()
axcomp.legend()
plt.savefig(os.path.join(output_dir, "racelines.eps"))
plt.savefig(os.path.join(output_dir, "racelines.svg"))
plt.savefig(os.path.join(output_dir, "racelines.png"))
distances = distancespoly
meandist = meandistpoly
stddist = stdevpoly
histfig = plt.figure()
num_bins = 30
n, bins, patches = plt.hist(distances, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel("Distance to Optimal Raceline")
plt.ylabel("Number of Samples")
# dmax = meandist+2*stddist
# N = 1000
# kdexplot = np.linspace(0,dmax,N).reshape(-1, 1)

# font = {#'family' : 'normal',
#         #'weight' : 'bold',
#         'size'   : 15}

# plt.rc('font', **font)

# figkde, axkde = plt.subplots()
# figkde.subplots_adjust(hspace=0.05, wspace=0.05)
# kernel='gaussian'
# kde = KernelDensity(kernel=kernel, bandwidth=0.25)
# kde.fit(distances.reshape(-1, 1))
# log_dens = kde.score_samples(kdexplot)
# pdf = np.exp(log_dens)
# axkde.plot(np.hstack((np.array([0]),kdexplot[:,0])), np.hstack((np.array([0]),pdf)), '-', label="kernel = '{0}'".format(kernel))
# axkde.set_xlabel("Minimum distance (m) to reference raceline")
# axkde.set_ylabel("Probability Density")
plt.savefig(os.path.join(output_dir, "histogram.eps"))
plt.savefig(os.path.join(output_dir, "histogram.svg"))
plt.savefig(os.path.join(output_dir, "histogram.png"))


# figkdevel, axkdevel = plt.subplots()
# figkdevel.subplots_adjust(hspace=0.05, wspace=0.05)
# kernel='gaussian'
# kdevel = KernelDensity(kernel=kernel, bandwidth=0.25).fit(speed_diffs.reshape(-1, 1))
# log_densvel = kdevel.score_samples(kdevelxplot)
# pdfvel = np.exp(log_densvel)
# axkdevel.plot(kdevelxplot[:,0], pdfvel, '-', label="kernel = '{0}'".format(kernel))
# axkdevel.set_xlabel("Difference in speed (m/s) from closest point in reference raceline")
# axkdevel.set_ylabel("Probability Density")



# figveltrace, axveltrace = plt.subplots()
# veltrace = np.loadtxt(velocity_trace,delimiter=",")
# tveltrace = np.linspace(0,veltrace.shape[0]-1,veltrace.shape[0])/60.0
# axveltrace.plot(tveltrace,veltrace[:,0], '-', label="setpoint", color="r")
# axveltrace.plot(tveltrace, veltrace[:,1], '-', label="actual", color="g")
# axveltrace.legend()
# axveltrace.set_xlabel("Time")
# axveltrace.set_ylabel("Speed (m/s)")





plt.show(block=True)

