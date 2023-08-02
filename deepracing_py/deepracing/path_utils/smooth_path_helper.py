import typing
import numpy as np
from typing import List, Union, Tuple
from scipy.spatial.transform import Rotation, RotationSpline
import shapely.geometry
import numpy as np
import scipy.interpolate
import scipy.integrate
import scipy.linalg
from scipy.spatial import KDTree
from scipy.optimize import minimize_scalar
import scipy.optimize

def generateSpline(points : np.ndarray, k=3, simpson_subintervals = 12):
    if not (simpson_subintervals%2)==0:
        raise ValueError("Must use even number of subintervals for Simpson's method")
    # points_aug : np.ndarray = points[:-1].copy()
    if np.linalg.norm(points[-1]-points[0], ord=2)>1E-6:
        points_aug = np.concatenate([points, points[None,0]], axis=0)
    else:
        points_aug = points.copy()
    euclidean_distances : np.ndarray = np.zeros_like(points_aug[:,0])
    euclidean_distances[1:] = np.cumsum(np.linalg.norm(points_aug[1:] - points_aug[:-1], ord=2, axis=1))
    splinex = euclidean_distances

    fake_spline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(splinex, points_aug, k=k, bc_type="periodic")
    fake_spline_der : scipy.interpolate.BSpline = fake_spline.derivative()

    true_distances : np.ndarray = np.zeros_like(splinex)
    for i in range(1, true_distances.shape[0]):
        x0 = splinex[i-1]
        x1 = splinex[i]
        xvec : np.ndarray = np.linspace(x0, x1, simpson_subintervals + 1)
        spline_norms : np.ndarray = np.linalg.norm(fake_spline_der(xvec), axis=1)
        true_distances[i] = true_distances[i-1] + scipy.integrate.simpson(spline_norms, xvec)
    return true_distances, points_aug, scipy.interpolate.make_interp_spline(true_distances, points_aug, k=k, bc_type="periodic")
def generateTimestamps(distances : np.ndarray, speeds : np.ndarray):
    times : np.ndarray = np.zeros_like(distances)
    for i in range(1, times.shape[0]):
        v0 = float(speeds[i-1])
        vf = float(speeds[i])
        dr = float(distances[i] - distances[i-1])
        a0 = 0.5*(vf**2 - v0**2)/dr
        poly : np.polynomial.Polynomial = np.polynomial.Polynomial([-dr, v0, 0.5*a0])
        roots : np.ndarray = poly.roots()
        realroots = roots[np.abs(roots.imag)<1E-4].real
        positiverealroots = realroots[realroots>0.0]
        dt = np.min(positiverealroots)
        times[i] = times[i-1] + dt
    return times

class SmoothPathHelper:
    def __init__(self, points : np.ndarray, speeds : typing.Union[np.ndarray, None] = None, k=3, simpson_subintervals = 12):
        true_distances, points_aug, spline = generateSpline(points, k=k, simpson_subintervals=simpson_subintervals)
        self.distances : np.ndarray = true_distances
        self.points : np.ndarray = points_aug
        self.spline : scipy.interpolate.BSpline = spline
        self.spline_derivative : scipy.interpolate.BSpline = self.spline.derivative()
        self.spline_2nd_derivative : scipy.interpolate.BSpline = self.spline.derivative(nu=2)
        self.ring : shapely.geometry.LinearRing = shapely.geometry.LinearRing(self.points)
        self.polygon : shapely.geometry.Polygon = shapely.geometry.Polygon(self.ring)
        self.kdtree : KDTree = KDTree(self.points)
        self.k : int = k

        if speeds is not None:
            self.parameterize_time(speeds)
        else:
            self.speeds = None
            self.times = None
            self.spline_time = None
            self.spline_time_derivative = None
            self.spline_time_2nd_derivative = None
            self.speed_of_r = None
            self.r_of_t = None
            self.t_of_r = None

    def parameterize_time(self, speeds : np.ndarray):
        self.speeds : np.ndarray = np.zeros_like(self.distances)
        self.speeds[:-1] = speeds.copy()
        self.speeds[-1] = self.speeds[0]
        self.times = generateTimestamps(self.distances, self.speeds)
        self.spline_time : scipy.interpolate.BSpline =  scipy.interpolate.make_interp_spline(self.times, self.points, k=self.k, bc_type="periodic")
        self.spline_time_derivative : scipy.interpolate.BSpline = self.spline_time.derivative()
        self.spline_time_2nd_derivative : scipy.interpolate.BSpline = self.spline_time.derivative(nu=2)
        # self.speed_of_r : scipy.interpolate.Akima1DInterpolator = scipy.interpolate.Akima1DInterpolator(self.distances, self.speeds)
        self.speed_of_r : scipy.interpolate.BSpline =  scipy.interpolate.make_interp_spline(self.distances, self.speeds, k=self.k, bc_type="periodic")
        self.r_of_t : scipy.interpolate.Akima1DInterpolator = scipy.interpolate.Akima1DInterpolator(self.times, self.distances)
        self.t_of_r : scipy.interpolate.Akima1DInterpolator = scipy.interpolate.Akima1DInterpolator(self.distances, self.times)

    def __closest_point_functor__(self, r : typing.Union[float,np.ndarray], query_point : np.ndarray, hession_mat : np.ndarray):
        if hession_mat is None:
            return np.linalg.norm(self.spline(r) - query_point, ord=2)
        deltas = self.spline(r) - query_point
        squared_deltas = np.square(deltas)
        grad = np.sum(deltas*self.spline_derivative(r), axis=1)
        scale = float(1.0/deltas.shape[0])
        return 0.5*scale*np.sum(squared_deltas), scale*grad
    def __closest_point_hessian(self, r : np.ndarray, query_point : np.ndarray, hession_mat : np.ndarray):
        return hession_mat
    def closest_point(self, query_point : np.ndarray, bounds_delta=(3.0, 3.0) ) -> typing.Tuple[typing.Union[float,np.ndarray], np.ndarray]:
        _, iclosest = self.kdtree.query(query_point)
        rguess = self.distances[iclosest]
        if query_point.ndim==2:
            for i in range(0, rguess.shape[0]-1):
                if rguess[i+1]<rguess[i]:
                    rguess[i+1:]+=self.distances[-1]
                    rguess-=self.distances[-1]
                    break
        if query_point.ndim==1:
            bounds = (rguess - bounds_delta[0], rguess + bounds_delta[1])
            res = minimize_scalar(self.__closest_point_functor__, bounds=bounds, args=(query_point, None), method="bounded")
            rout : float = float(res.x)
        elif query_point.ndim==2:
            bounds : scipy.optimize.Bounds = scipy.optimize.Bounds(rguess - bounds_delta[0], rguess + bounds_delta[1])
            hession_mat : np.ndarray = np.zeros([rguess.shape[0], rguess.shape[0]], dtype=rguess.dtype)
            res = scipy.optimize.minimize(self.__closest_point_functor__, rguess,\
                                           hess=self.__closest_point_hessian, bounds=bounds, tol=1E-8, \
                                            jac=True, args=(query_point,hession_mat), method="SLSQP")
            rout : np.ndarray = res.x
        # delta = rguess - rout
        # if np.abs(delta)>bounds_delta[0] or np.abs(delta)>bounds_delta[1]:
        #     raise ValueError( "WHOA THERE. DELTA IS WAAAY TOO BIG: %f" % (delta,) )
        return rout, self.spline(rout)
    def __normal_projection_functor__(self, r : float, point_on_reference : np.ndarray, heading_vector : np.ndarray):
        return np.abs( np.dot( self.spline(r) - point_on_reference , heading_vector ) )
    def normal_projection(self, r : float, other_path : 'SmoothPathHelper', bounds = None , guess = None):
        point_on_this : np.ndarray = self.spline(r)
        heading_on_this : np.ndarray = self.spline_derivative(r)
        if guess is None:
            _, iclosest = other_path.kdtree.query(point_on_this)
            # euclidean_projection : float = r
            euclidean_projection : float = other_path.distances[iclosest]
            if(euclidean_projection>r+other_path.distances[-1]/2.0):
                euclidean_projection-=other_path.distances[-1]
            rguess = euclidean_projection
        else:
            rguess = guess
        if bounds is None:
            bounds_ = np.asarray([rguess - 8.0, rguess + 8.0])
        else:
            bounds_ = np.asarray([rguess - bounds[0], rguess + bounds[1]])
        # bounds_ = np.clip(bounds_, other_path.distances[0] - 1.0, other_path.distances[-1] + 1.0)
        res = minimize_scalar(other_path.__normal_projection_functor__, bounds=(bounds_[0], bounds_[1]), args=(point_on_this, heading_on_this), method="bounded")
        rout = res.x
        # rout = rout%other_path.distances[-1]
        return rout, other_path.spline(rout)
    def point(self, r) -> np.ndarray:
        return self.spline(r)
    def direction(self, r) -> np.ndarray:
        return self.spline_derivative(r)
    def normal(self, r) -> np.ndarray:
        tangent = self.direction(r)
        return np.asarray([-tangent[1], tangent[0]])
    def __call__(self, r : Union[float, np.ndarray]):
        point = self.spline(r)
        direction = self.spline_derivative(r)
        normal = self.spline_2nd_derivative(r)
        if type(r)==float:
            rotation : Rotation = Rotation.from_rotvec([0.0, 0.0, np.arctan2(direction[1], direction[0])])
            pose : np.ndarray = np.eye(3, dtype=point.dtype)
            pose[0:2,0:2] = rotation.as_matrix().astype(point.dtype)[0:2,0:2]
            pose[0:2,2] = point
            curvature = np.linalg.norm(normal, ord=2)
            return pose, curvature
        elif type(r)==np.ndarray:
            rotvecs = np.zeros((r.shape[0], 3), dtype=r.dtype)
            rotvecs[:,2] = np.arctan2(direction[:,1], direction[:,0])
            rotations : Rotation = Rotation.from_rotvec(rotvecs)
            poses : np.ndarray = np.zeros((rotvecs.shape[0], 3, 3), dtype=rotvecs.dtype)
            poses[:,2,2] = 1.0
            poses[:,0:2,0:2] = rotations.as_matrix()[:,0:2,0:2]
            poses[:,0:2,2] = point 
            curvatures = np.linalg.norm(normal, ord=2, axis=1)
            return poses, curvatures
        else:
            raise ValueError("r must be either a float or 1-d np.ndarray")
