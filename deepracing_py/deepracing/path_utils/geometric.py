import numpy as np
import scipy
import scipy.integrate as integrate
import scipy.interpolate as interp
class SplineSpeedWrapper():
    def __init__(self, velspline : interp.BSpline):
        self.velspline = velspline
    def func(self, s):
        return np.linalg.norm(self.velspline(s), ord=2)



def computeTangentsAndNormals(r, X, k = 3, ds = 1.0, ref = np.array([0.0,1.0,0.0], dtype=np.float64)):
    
    finaldelta = np.linalg.norm(X[-1] - X[0], ord=2)
    Xaug = np.concatenate([X, X[0].reshape(1,-1)], axis=0)
    raug = np.concatenate([r, np.asarray([r[-1]+finaldelta])], axis=0)
    raug = raug-raug[0]
    garbagespline : interp.BSpline = interp.make_interp_spline(raug, Xaug, k=k, bc_type="periodic")
    garbagesplineder : interp.BSpline = garbagespline.derivative()
    truedistances : np.ndarray = np.zeros_like(raug)
    for i in range(1, truedistances.shape[0]):
        rsumsamp : np.ndarray = np.linspace(raug[i-1], raug[i], num=5)
        velsubsamp : np.ndarray = np.linalg.norm(garbagesplineder(rsumsamp), ord=2, axis=1)
        truedistances[i] = truedistances[i-1] + integrate.simpson(velsubsamp, x=rsumsamp)

    spline : interp.BSpline = interp.make_interp_spline(truedistances, Xaug, k=k, bc_type="periodic")
    tangentspline : interp.BSpline = spline.derivative(nu=1)
    rsamp = np.linspace(truedistances[0], truedistances[-1]-ds, num = int(round(truedistances[-1]/ds)))
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


    

