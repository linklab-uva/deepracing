import numpy as np
import scipy
import scipy.integrate as integrate
import scipy.interpolate as interp


def computeTangentsAndNormals(r, X, k = 3, ds = 1.0, ref = np.array([0.0,1.0,0.0], dtype=np.float64)):
    finaldelta = np.linalg.norm(X[-1] - X[0], ord=2)
    Xaug = np.concatenate([X, X[0].reshape(1,-1)], axis=0)
    raug = np.concatenate([r, np.asarray([r[-1]+finaldelta])], axis=0)
    spline : interp.BSpline = interp.make_interp_spline(raug, Xaug, k=k, bc_type="periodic")
    tangentspline : interp.BSpline = spline.derivative(nu=1)
    rsamp = np.arange(raug[0], raug[-1]-ds, step = ds)
    points = spline(rsamp)
    tangents = tangentspline(rsamp)
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

    return spline, points, speeds, unit_tangents, unit_normals, rsamp


    

