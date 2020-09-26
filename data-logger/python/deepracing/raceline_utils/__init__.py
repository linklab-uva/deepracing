import numpy as np
import scipy.interpolate
from scipy.interpolate import make_interp_spline, BSpline, CubicSpline
from scipy.spatial.transform import Rotation as Rot

def shiftRaceline(raceline: np.ndarray, reference_vec: np.ndarray, distance: float, s = None):
    diffs = raceline[1:] - raceline[0:-1]
    diffnorms = np.linalg.norm(diffs, axis=1, ord=2)
    if s is None:
        s_ = np.hstack([np.zeros(1), np.cumsum(diffnorms)])
    else:
        assert(s.shape[0]==raceline.shape[0])
        s_ = s.copy()
    nogood = diffnorms==0.0
    good = np.hstack([np.array([True]), ~nogood])
    s_ = s_[good]
    racelinegood = raceline[good].copy()
    racelinespl : CubicSpline = CubicSpline(s_, racelinegood)
    racelinetanspl = racelinespl.derivative()
    tangents = racelinetanspl(s_)
    tangent_norms = np.linalg.norm(tangents,axis=1,ord=2)
    unit_tangents = tangents/tangent_norms[:,np.newaxis]
    laterals = np.row_stack([np.cross(unit_tangents[i], reference_vec) for i in range(unit_tangents.shape[0])])
    lateral_norms = np.linalg.norm(laterals,axis=1,ord=2)
    unit_laterals = laterals/lateral_norms[:,np.newaxis]
    return s_, racelinegood + distance*unit_laterals