import numpy as np
import scipy
import scipy.integrate as integrate
import scipy.interpolate as interp


def computeTangentsAndNormals(r, X, k = 3, rsamp = None, ref = np.array([0.0,1.0,0.0], dtype=np.float64), bc_type = None):
    spline : interp.BSpline = interp.make_interp_spline(r, X, k=k, bc_type=bc_type)
    tangentspline : scipy.interpolate.BSpline = spline.derivative(nu=1)
    if rsamp is None:
        rsamp_ = r.copy()
    else:
        rsamp_ = rsamp.copy()
    points = spline(rsamp_)
    tangents = tangentspline(rsamp_)
    speeds = np.linalg.norm(tangents, ord=2, axis=1)
    unit_tangents = tangents/speeds[:,np.newaxis]

    numpoints = points.shape[0]
    ref_ = np.stack([ref for asdf in range(numpoints)])
    v1 = np.cross(unit_tangents, ref_)
    v1 = v1/np.linalg.norm(v1, axis=1, ord=2)[:,np.newaxis]
    v2 =  np.cross(v1, unit_tangents)
    v2 = v2/np.linalg.norm(v2, axis=1, ord=2)[:,np.newaxis]

    normals = np.cross(v2, unit_tangents)
    unit_normals = normals/np.linalg.norm(normals, axis=1, ord=2)[:,np.newaxis]

    return spline, points, speeds, unit_tangents, unit_normals


    

