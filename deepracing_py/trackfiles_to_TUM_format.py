import numpy as np
from PIL import Image as PILImage
import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import deepracing
import sklearn.decomposition
import scipy.linalg
from scipy.spatial.transform import Rotation as Rot
import sklearn.neighbors


parser = argparse.ArgumentParser()
parser.add_argument("track_name", help="Path to trackfile to convert",  type=str)
#parser.add_argument("ds", type=float, help="Sample the boundaries at points this distance apart")
parser.add_argument("--track-dirs", help="Directories to search for track files, defaults to value of F1_TRACK_DIRS environment variable",  type=str, nargs="+", default=None)
parser.add_argument("--out", help="where to put the output file",  type=str, default=os.path.curdir)
args = parser.parse_args()
argdict = vars(args)

outdir = argdict["out"]
track_name = argdict["track_name"]
track_dirs = argdict["track_dirs"]
if track_dirs is None:
    track_dirs_env = os.getenv("F1_TRACK_DIRS", None)
    if track_dirs_env is None:
        raise ValueError("Must either set F1_TRACK_DIRS environment variable or explicity specify --track-dirs")
    track_dirs = str.split(track_dirs_env, os.pathsep)

inner_boundary_file = deepracing.searchForFile(track_name+"_innerlimit.json", track_dirs)
with open(inner_boundary_file,"r") as f:
    inner_boundary_dict = json.load(f)
inner_boundary_in = np.column_stack([inner_boundary_dict["x"], inner_boundary_dict["y"], inner_boundary_dict["z"]])


outer_boundary_file = deepracing.searchForFile(track_name+"_outerlimit.json", track_dirs)
with open(outer_boundary_file,"r") as f:
    outer_boundary_dict = json.load(f)
outer_boundary_in = np.column_stack([outer_boundary_dict["x"], outer_boundary_dict["y"], outer_boundary_dict["z"]])


all_points = np.concatenate([inner_boundary_in, outer_boundary_in], axis=0)
minx = np.min(all_points[:,0])
maxx = np.max(all_points[:,0])
stdevy = np.std(all_points[:,1])
meany = np.mean(all_points[:,1])
print("Standard Deviation in y: " + str(stdevy))

boundary_pca = sklearn.decomposition.TruncatedSVD(n_components=2)
boundary_pca.fit(all_points)
print("PCA Singular Vectors: \n" + str(boundary_pca.components_))
print("Explain Variance Ratios: " + str(boundary_pca.explained_variance_ratio_))

inner_boundary_projected = boundary_pca.inverse_transform(boundary_pca.transform(inner_boundary_in))
outer_boundary_projected = boundary_pca.inverse_transform(boundary_pca.transform(outer_boundary_in))
# inner_boundary_projected = np.column_stack([inner_boundary_in[:,0], meany*np.ones_like(inner_boundary_in[:,0]), inner_boundary_in[:,2]])
# outer_boundary_projected = np.column_stack([outer_boundary_in[:,0], meany*np.ones_like(outer_boundary_in[:,0]), outer_boundary_in[:,2]])
all_points_projected = np.concatenate([inner_boundary_projected, outer_boundary_projected], axis=0)


Rmat = np.eye(3)
Rmat = scipy.linalg.orthogonal_procrustes(all_points, all_points_projected)[0] #[0:2,0:2] [:,[0,2]]
rot = Rot.from_matrix(Rmat)
rotvec = rot.as_rotvec()
print("Rotation vector from OPA: " + str(rotvec))
inner_boundary_projected = np.matmul(Rmat, inner_boundary_projected.T).T
outer_boundary_projected = np.matmul(Rmat, outer_boundary_projected.T).T


fig = plt.figure()

plt.plot(inner_boundary_in[:,0], inner_boundary_in[:,2],c="g")
plt.plot(outer_boundary_in[:,0], outer_boundary_in[:,2],c="g")
plt.plot(inner_boundary_projected[:,0], inner_boundary_projected[:,2], c="r")
plt.plot(outer_boundary_projected[:,0], outer_boundary_projected[:,2], c="r")
plt.plot([],[], c="g", label="Input Boundary")
plt.plot([],[], c="r", label="PCA-Transformed Boundary")
plt.xlim(maxx+5.0, minx-5.0)
ibxy = inner_boundary_projected[:,[0,2]]
obxy = outer_boundary_projected[:,[0,2]]
obkdtree = sklearn.neighbors.KDTree(obxy)
nearest_distances, nearest_ind = obkdtree.query(ibxy) 
nearest_ind = nearest_ind[:,0]
print(nearest_ind)
midpoints = (ibxy + obxy[nearest_ind])/2.0
plt.plot(midpoints[:,0], midpoints[:,1], c="b", label="midpoints")
plt.legend()
plt.show()


widths = 0.975*(nearest_distances/2.0)
# x_m,y_m,w_tr_right_m,w_tr_left_m

tumformat = np.column_stack([midpoints[:,0], midpoints[:,1], widths, widths])
outfile = os.path.join(outdir, track_name+".csv")
with open(outfile, "w") as f:
    np.savetxt(f,tumformat, fmt="%.6f", delimiter=",", header="# x_m,y_m,w_tr_right_m,w_tr_left_m")