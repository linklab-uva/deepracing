import deepracing.path_utils, deepracing
import argparse
import numpy as np
import scipy.interpolate, scipy.integrate
from scipy.spatial.transform import Rotation
import os
import yaml
import matplotlib.figure
import matplotlib.pyplot as plt

def go(argdict : dict):
    trackmapdir = argdict["trackmap"]

    innerboundfile = os.path.join(trackmapdir, "inner_boundary.pcd")
    innerboundtype, innerbound_structured, _, _ = deepracing.path_utils.loadPCD(innerboundfile, align=True)
    innerbound_lapdistance : np.ndarray = np.squeeze(innerbound_structured["lapdistance"])
    innerbound : np.ndarray = np.squeeze(np.stack([innerbound_structured["x"], innerbound_structured["y"], innerbound_structured["z"]], axis=1))

    outerboundfile = os.path.join(trackmapdir, "outer_boundary.pcd")
    outerboundtype, outerbound_structured, _, _ = deepracing.path_utils.loadPCD(outerboundfile, align=True)
    outerbound_lapdistance : np.ndarray = np.squeeze(outerbound_structured["lapdistance"])
    outerbound : np.ndarray = np.squeeze(np.stack([outerbound_structured["x"], innerbound_structured["y"], outerbound_structured["z"]], axis=1))
    
    k = argdict["k"]
    innerbound_lapdistance_spline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(innerbound_lapdistance, innerbound, k=k, bc_type="periodic")
    outerbound_lapdistance_spline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(outerbound_lapdistance, outerbound, k=k, bc_type="periodic")

    with open(os.path.join(trackmapdir, "metadata.yaml"), "r") as f:
        metadatadict : dict = yaml.load(f, Loader=yaml.SafeLoader)
    tracklength : float = metadatadict["tracklength"]
    
    rsamp : np.ndarray = np.arange(0.0, tracklength, step=argdict["dr"])
    
    ibsamp = innerbound_lapdistance_spline(rsamp)
    obsamp = outerbound_lapdistance_spline(rsamp)

    all_points = np.concatenate([ibsamp, obsamp], axis=0)

    ymean = np.mean(all_points[:,1])

    ibsamp[:,1] = ymean
    obsamp[:,1] = ymean

    centerline = 0.5*(ibsamp + obsamp)
    deltavecs = obsamp - ibsamp
    unitdeltavecs = deltavecs/np.linalg.norm(deltavecs, ord=2, axis=1)[:,np.newaxis]

    outline = centerline + argdict["offset"]*unitdeltavecs


    fig : matplotlib.figure.Figure = plt.figure()
    plt.plot(innerbound[:,0], innerbound[:,2], label="Inner Bound")
    plt.plot(outerbound[:,0], outerbound[:,2], label="Outer Bound")
    plt.plot(outline[:,0], outline[:,2], label="Output")
    plt.legend()
    plt.show()

    viewpoint_pos : np.ndarray = np.asarray(metadatadict["startingline_pose"]["position"])
    viewpoint_quat : np.ndarray = np.asarray(metadatadict["startingline_pose"]["quaternion"])
    viewpoint_rot : Rotation = Rotation.from_quat(viewpoint_quat)
    deepracing.path_utils.numpyToPCD(rsamp, outline, argdict["outfile"], x_name="lapdistance", viewpoint_pos=viewpoint_pos, viewpoint_rot=viewpoint_rot)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trackmap",  type=str, help="Path to .pcd file to generate optimal line from")
    parser.add_argument("--k", type=int, default=3, help="Degree of splines to use")
    parser.add_argument("--dr", type=float, default=1.0, help="Spacing to sample the boundary lines")
    parser.add_argument("--offset", type=float, default=0.0, help="Offset from the centerline to use")
    parser.add_argument("--outfile", type=str, default="offset_line.pcd", help="Where to put the resulting pcd")
    args = parser.parse_args()
    go(vars(args))