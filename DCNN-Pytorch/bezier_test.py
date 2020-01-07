import scipy
import scipy.integrate
import scipy.interpolate
from scipy.interpolate import make_interp_spline as mkspl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import array, linalg, matrix
from scipy.special import comb as nOk
import deepracing_models.math_utils
import torch
def lsqfit(points,M):
    M_ = linalg.pinv(M)
    return M_ * points
pi = np.pi
num_points = 1000
tmax = 10.0
kbezier = 5
d = 5
numcurves = 8
Mtk = lambda i, n, t: t**(i)*(1-t)**(n-i)*nOk(n,i)
bezierFactors = lambda ts, n: matrix([[Mtk(k_,n,t) for k_ in range(n+1)] for t in ts])
P = np.zeros((numcurves,num_points,2))
Pprime = np.zeros((numcurves,num_points,2))
tnp = np.zeros((numcurves,num_points))
for i in range(numcurves):
   #P[i] = np.zeros((num_points,2))
    t = np.linspace(0, tmax*np.random.rand(), num_points )
    tnp[i] = t.copy()
    p = 5.0*(np.random.rand(d)-np.random.rand(d))
    tfunc = t-t[int(len(t)/2)]
    P[i,:,0] = tnp[i]#*np.sin(tnp[i])# - tnp[i]**2
    P[i,:,1] = np.polyval(p,t)# + 10.0*(t**2)*np.cos(t)
    P[i,:,1] = P[i,:,1] - P[i,0,1] 
    Pprime[i,:,0] = np.ones(tfunc.shape[0])
    Pprime[i,:,1] = np.polyval(np.polyder(p),t)# + 10.0*((t**2)*(-np.sin(t)) + 2*t*np.cos(t))

Ptorch = torch.from_numpy(P.copy()).double()#.unsqueeze(0)
Pprimetorch = torch.from_numpy(Pprime.copy()).double()#.unsqueeze(0)
ttorch = torch.from_numpy(tnp.copy()).double()#.unsqueeze(0)
dt = ttorch[:,-1]-ttorch[:,0]
storch = (ttorch - ttorch[:,0,None])/dt[:,None]
#ttorch = ttorch.repeat(numcurves,1)
M , bezier_control_points = math_utils.bezierLsqfit(Ptorch,storch,kbezier)
print(bezier_control_points.shape)
Pbeziertorch = torch.matmul(M,bezier_control_points)
Mderiv, Pbeziertorchderiv = math_utils.bezierDerivative(bezier_control_points,storch)
Pbeziertorchderiv= Pbeziertorchderiv/dt[:,None,None]

for i in range(numcurves):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(P[i,:,0],P[i,:,1],'r-')
    ax.scatter(Pbeziertorch[i,:,0].numpy(),Pbeziertorch[i,:,1].numpy(), facecolors='none', edgecolors='b')
    skipn = 20
    #ax.quiver(Pbeziertorch[::skipn,0].numpy(),Pbeziertorch[::skipn,1].numpy(),Pbeziertorchderiv[::skipn,0].numpy(),Pbeziertorchderiv[::skipn,1].numpy())
    ax.plot(bezier_control_points[i,:,0].numpy(),bezier_control_points[i,:,1].numpy(),'go')
    print("Mean distance: %f" %(torch.mean(torch.norm(Pbeziertorch[i]-Ptorch[i],dim=1,p=2)).item()))
    print("Mean velocity diff: %f" %(torch.mean(torch.norm(Pbeziertorchderiv[i]-Pprimetorch[i],dim=1,p=2)).item()))
    plt.show()