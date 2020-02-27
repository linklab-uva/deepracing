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
tmin = -2.0
tmax = 2.0
d = 3
kbezier = d - 0
numcurves = 8
Mtk = lambda i, n, t: t**(i)*(1-t)**(n-i)*nOk(n,i)
bezierFactors = lambda ts, n: matrix([[Mtk(k_,n,t) for k_ in range(n+1)] for t in ts])
P = np.zeros((numcurves,num_points,2))
Pprime = np.zeros((numcurves,num_points,2))
Pprimeprime = np.zeros((numcurves,num_points,2))
tnp = np.zeros((numcurves,num_points))
for i in range(numcurves):
   #P[i] = np.zeros((num_points,2))
    t = np.linspace(tmin*np.random.rand(), tmax*np.random.rand(), num_points )
    tfunc = t
   # tfunc = t-t[int(len(t)/2)]
    tnp[i] = tfunc.copy()
    px = 5.0*np.random.randn(d)
    py = 5.0*np.random.randn(d)
    P[i,:,0] = np.polyval(px,tnp[i])
    P[i,:,1] = np.polyval(py,tnp[i])
    P[i,:,0] = P[i,:,0] - P[i,0,0] 
    P[i,:,1] = P[i,:,1] - P[i,0,1] 
    Pprime[i,:,0] = np.polyval(np.polyder(px, m=1),tnp[i])
    Pprime[i,:,1] = np.polyval(np.polyder(py, m=1),tnp[i])
    Pprimeprime[i,:,0] = np.polyval(np.polyder(px, m=2),tnp[i])
    Pprimeprime[i,:,1] = np.polyval(np.polyder(py, m=2),tnp[i])

Ptorch = torch.from_numpy(P.copy()).double()#.unsqueeze(0)
Pprimetorch = torch.from_numpy(Pprime.copy()).double()#.unsqueeze(0)
Pprimeprimetorch = torch.from_numpy(Pprimeprime.copy()).double()#.unsqueeze(0)
ttorch = torch.from_numpy(tnp.copy()).double()#.unsqueeze(0)
dt = ttorch[:,-1]-ttorch[:,0]
storch = (ttorch - ttorch[:,0,None])/dt[:,None]
#ttorch = ttorch.repeat(numcurves,1)
print(storch)
print(Ptorch.shape)
M , bezier_control_points = math_utils.bezierLsqfit(Ptorch, kbezier, t = storch )
print(bezier_control_points.shape)
Pbeziertorch = torch.matmul(M,bezier_control_points)


Mderiv, Pdot_s = math_utils.bezierDerivative(bezier_control_points,storch, order=1)
Pdot_t= Pdot_s/dt[:,None,None]

Mdotdot, Pdotdot_s = math_utils.bezierDerivative(bezier_control_points, storch, order=2)
Pdotdot_t= Pdotdot_s/((dt**2)[:,None,None])


for i in range(numcurves):
    fig : Figure = plt.figure()
    ax : Axes = fig.add_subplot()
   #ax.plot(P[i,:,0],P[i,:,1],'r-')
    ax.set_title("A Bézier Curve and its Control Points")
    ax.plot(Pbeziertorch[i,:,0].numpy(),Pbeziertorch[i,:,1].numpy(), color='blue', label="A Bézier Curve")
   # ax.scatter(Pbeziertorch[i,:,0].numpy(),Pbeziertorch[i,:,1].numpy(), facecolors='none', edgecolors='b')
    skipn = 50
    #ax.plot(bezier_control_points[i,:,0].numpy(),bezier_control_points[i,:,1].numpy(),'go')
    ax.scatter(bezier_control_points[i,:,0].numpy(),bezier_control_points[i,:,1].numpy(), facecolors='none', edgecolors='g', label="Control Points")
    ax.quiver(Pbeziertorch[i,::skipn,0].numpy(),Pbeziertorch[i,::skipn,1].numpy(),Pdot_t[i,::skipn,0].numpy(),Pdot_t[i,::skipn,1].numpy(), angles='xy', color='black', label="Velocity Vectors")
    ax.legend(loc="best")

    print("Mean distance: %f" %(torch.mean(torch.norm(Pbeziertorch[i]-Ptorch[i],dim=1,p=2)).item()))
    print("Mean velocity diff: %f" %(torch.mean(torch.norm(Pdot_t[i]-Pprimetorch[i],dim=1,p=2)).item()))
    print("Mean acceleration diff: %f" %(torch.mean(torch.norm(Pdotdot_t[i]-Pprimeprimetorch[i],dim=1,p=2)).item()))
    plt.show()