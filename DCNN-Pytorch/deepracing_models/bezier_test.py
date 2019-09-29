import scipy
import scipy.integrate
import scipy.interpolate
from scipy.interpolate import make_interp_spline as mkspl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import bezier.curve
from numpy import array, linalg, matrix
from scipy.special import comb as nOk
import math_utils
import torch
def lsqfit(points,M):
    M_ = linalg.pinv(M)
    return M_ * points
pi = np.pi
num_points = 1000
tmax = 1.0
kbezier = 7
d = 5
numcurves = 8
Mtk = lambda i, n, t: t**(i)*(1-t)**(n-i)*nOk(n,i)
bezierFactors = lambda ts, n: matrix([[Mtk(k_,n,t) for k_ in range(n+1)] for t in ts])
P = np.zeros((numcurves,num_points,2))
tnp = np.zeros((numcurves,num_points))
for i in range(numcurves):
   #P[i] = np.zeros((num_points,2))
    t = np.linspace(-np.random.rand(), tmax + np.random.rand()*2.0 , num_points )
    tnp[i] = ((t-t[0])/(t[-1]-t[0])).copy()
    p = 5.0*(np.random.rand(d)-np.random.rand(d))
    tfunc = t-t[int(len(t)/2)]
    P[i,:,0] = tnp[i]#*np.sin(tnp[i])# - tnp[i]**2
    P[i,:,1] = np.polyval(p,tfunc)# + 10.0*t*np.cos(t)
    P[i,:,1] = P[i,:,1] - P[i,0,1] 

Ptorch = torch.from_numpy(P.copy()).double()#.unsqueeze(0)
ttorch = torch.from_numpy(tnp.copy()).double()#.unsqueeze(0)
#ttorch = ttorch.repeat(numcurves,1)
M , bezier_control_points = math_utils.bezierLsqfit(Ptorch,ttorch,kbezier)
print(bezier_control_points.shape)
Pbeziertorch = torch.matmul(M,bezier_control_points)
Pbeziertorchderiv = math_utils.bezierDerivative(bezier_control_points,kbezier,ttorch)

for i in range(numcurves):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(P[i,:,0],P[i,:,1],'r-')
    ax.plot(Pbeziertorch[i,:,0].numpy(),Pbeziertorch[i,:,1].numpy(),'bo')
    skipn = 20
    #ax.quiver(Pbeziertorch[::skipn,0].numpy(),Pbeziertorch[::skipn,1].numpy(),Pbeziertorchderiv[::skipn,0].numpy(),Pbeziertorchderiv[::skipn,1].numpy())
    ax.plot(bezier_control_points[i,:,0].numpy(),bezier_control_points[i,:,1].numpy(),'go')
    plt.show()