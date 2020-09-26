import numpy as np
import scipy.interpolate
from scipy.interpolate import make_interp_spline, BSpline
from scipy.spatial.transform import Rotation as Rot

def shiftRaceline(raceline: np.ndarray, reference_vec: np.ndarray, distance: float, s = None, k = 3):
    if s is None:
        diffs = raceline[1:] - raceline[0:-1]
        diffnorms = np.linalg.norm(diffs, axis=1, ord=2)
        s_ = np.hstack([np.zeros(1), np.cumsum(diffnorms)])
    else:
        s_ = s
    racelinespl : BSpline = make_interp_spline(s_, raceline, k = k)
    racelinetanspl = racelinespl.derivative()
    tangents = racelinetanspl(s_)
    tangent_norms = np.linalg.norm(tangents,axis=1,ord=2)
    unit_tangents = tangents/tangent_norms[:,np.newaxis]
    laterals = np.row_stack([np.cross(unit_tangents[i], reference_vec) for i in range(unit_tangents.shape[0])])
    lateral_norms = np.linalg.norm(laterals,axis=1,ord=2)
    unit_laterals = laterals/lateral_norms[:,np.newaxis]
    return s_, raceline + distance*unit_laterals