import numpy as np
from scipy.spatial.transform import Rotation

from .pcd_utils import loadPCD, decodePCDHeader, numpyToPCD, structurednumpyToPCD

from .smooth_path_helper import SmoothPathHelper


def paramaterize_time(speeds : np.ndarray, arclengths : np.ndarray):
    times = np.zeros_like(speeds)
    for i in range(1, times.shape[0]):
        dr = arclengths[i] - arclengths[i-1]
        v0 = speeds[i-1]
        vf = speeds[i]
        accel = (vf**2 - v0**2)/(2.0*dr)
        poly : np.polynomial.Polynomial = np.polynomial.Polynomial([-dr, v0, 0.5*accel])
        roots : np.ndarray = np.real(poly.roots())
        positiveroots = roots[roots>0.0]
        times[i] = times[i-1] + float(np.min(positiveroots))
    return times