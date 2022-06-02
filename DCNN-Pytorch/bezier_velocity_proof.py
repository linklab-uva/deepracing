import scipy
import scipy.integrate
import scipy.interpolate
from scipy.interpolate import make_interp_spline as mkspl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from numpy import array, linalg, matrix
from scipy.special import comb as nOk
import deepracing_models.math_utils as math_utils
import torch
def lsqfit(points,M):
    M_ = linalg.pinv(M)
    return M_ * points
pi = np.pi
num_points = 1000
tmin = -1.0
tmax = 1.0
kbezier = 3
t = torch.linspace(0.0, 2.25, dtype=torch.float64, steps=100)
#t,_ = torch.sort(t + 0.1*torch.randn_like(t))
# t = t-t[0]
t0 = t[0]
tf = t[-1]
dt = tf-t0
s = (t-t0)/dt

px_t = torch.as_tensor([0.0, 0.0, 1.0, 0.0], dtype=t.dtype, device=t.device)
# px_t+=0.5*torch.randn_like(px_t)
py_t = torch.as_tensor([0.5, 1.0, 0.0, 0.0], dtype=t.dtype, device=t.device)
# py_t+=0.5*torch.randn_like(py_t)
points_true = torch.stack([torch.from_numpy(np.polyval(px_t.cpu().numpy(), t)), torch.from_numpy(np.polyval(py_t.cpu().numpy(), t))], dim=1)
Pt = torch.stack([px_t, py_t], dim=1)
bezier_to_poly, poly_to_bezier = math_utils.polynomialFormConversion(kbezier, dtype=t.dtype, device=t.device)
dP_dt_true = torch.as_tensor(np.stack([np.polyval(np.polyder(px_t.cpu().numpy()), t), np.polyval(np.polyder(py_t.cpu().numpy()), t)], axis=1)).type_as(t).to(t.device)
speeds_true = torch.norm(dP_dt_true, p=2, dim=1)

px_s = torch.as_tensor([px_t[0]*(dt**3), px_t[1]*(dt**2), px_t[2]*dt, px_t[3].item()], dtype=px_t.dtype, device=px_t.device)
py_s = torch.as_tensor([py_t[0]*(dt**3), py_t[1]*(dt**2), py_t[2]*dt, py_t[3].item()], dtype=py_t.dtype, device=py_t.device)
px_s = torch.flip(px_s, dims=[0])
py_s = torch.flip(py_s, dims=[0])
Ps = torch.stack([px_s, py_s], dim=1)
# bezierM, bezier_fit = math_utils.bezierLsqfit(points_true.unsqueeze(0), kbezier, t=s.unsqueeze(0))
# bezierM = bezierM[0]
# bezier_fit=bezier_fit[0]
bezier_fit = torch.matmul(poly_to_bezier, Ps)
bezierM = math_utils.bezierM(s.unsqueeze(0), kbezier)[0]
# print(t)
# print(bezier_fit.numpy())
# print(t[-1].item(), np.polyval(py_t.cpu().numpy(), t[-1]))
bezier_points = torch.matmul(bezierM, bezier_fit)

bezierMderiv, db_ds = math_utils.bezierDerivative(bezier_fit.unsqueeze(0), t=s.unsqueeze(0))
bezierMderiv = bezierMderiv[0]
db_ds=db_ds[0]

db_dt = db_ds/dt
speeds_bezier = torch.norm(db_dt, p=2, dim=1)
print(speeds_bezier - speeds_true)

figpoints, axpoints = plt.subplots()
plt.scatter(points_true[:,0], points_true[:,1], s=10.0, c="green")
plt.plot(bezier_points[:,0], bezier_points[:,1], c="blue")
plt.show()

