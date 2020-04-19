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
d = 4
kbezier = d - 0
numcurves = d
Mtk = lambda i, n, t: t**(i)*(1-t)**(n-i)*nOk(n,i)
bezierFactors = lambda ts, n: matrix([[Mtk(k_,n,t) for k_ in range(n+1)] for t in ts])
P = np.zeros((numcurves,num_points,2))
Pprime = np.zeros((numcurves,num_points,2))
Pprimeprime = np.zeros((numcurves,num_points,2))
tnp = np.zeros((numcurves,num_points))
t = np.linspace(tmin, tmax, num_points )
tfunc = t
for order in range(1,numcurves+1):
    i = order-1
   #P[i] = np.zeros((num_points,2))
   # tfunc = t-t[int(len(t)/2)]
    tnp[i] = tfunc.copy()
    px = np.zeros(order+1)
    px [-2] = 1
    py = np.zeros(order+1)
    py [0] = 1.0
    py [1] = -0.5
    if(order>2):
        py [2] = -0.5
    if(order>3):
        py [3] = 0.5
    py[-1]=0.0

   # py = 3.0*np.random.randn(order+1)
    P[i,:,0] = np.polyval(px,tnp[i])
    P[i,:,1] = np.polyval(py,tnp[i])
    # P[i,:,0] = P[i,:,0] - P[i,0,0] 
    # P[i,:,1] = P[i,:,1] - P[i,0,1] 
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

for i in range(numcurves):
    fig : Figure = plt.figure() 
    kbezier = i+1
    pt = Ptorch[i].unsqueeze(0)
    print(pt.shape)
    M , bezier_control_points = math_utils.bezierLsqfit(pt, kbezier, t = storch[0].unsqueeze(0) )
    print(bezier_control_points.shape)
    Pbeziertorch = torch.matmul(M,bezier_control_points)

    if kbezier>1:
        Mderiv, Pdot_s = math_utils.bezierDerivative(bezier_control_points,storch[0].unsqueeze(0), order=1)
        Pdot_t= Pdot_s/dt[i]
    else:
        vel = (bezier_control_points[0,1] - bezier_control_points[0,0])/dt[i]
        Pdot_t = torch.zeros_like(bezier_control_points)
        for i in range(Pdot_t.shape[1]):
            Pdot_t[0,i] = vel

    # Mdotdot, Pdotdot_s = math_utils.bezierDerivative(bezier_control_points, storch, order=2)
    # Pdotdot_t= Pdotdot_s/((dt**2)[:,None,None])

    #plt.plot(P[0,:,0],P[0,:,1],'r-')
    plt.title("A Bézier Curve of Order %d and its Control Points" % kbezier)
    plt.plot(Pbeziertorch[0,:,0].numpy(),Pbeziertorch[0,:,1].numpy(), color='blue', label="A Bézier Curve")

    #plt.scatter(Pbeziertorch[0,:,0].numpy(),Pbeziertorch[0,:,1].numpy(), facecolors='none', edgecolors='b', label="A Bézier Curve")
    skipn = 50
    #ax.plot(bezier_control_points[0,:,0].numpy(),bezier_control_points[0,:,1].numpy(),'go')
    plt.scatter(bezier_control_points[0,:,0].numpy(),bezier_control_points[0,:,1].numpy(), facecolors='none', edgecolors='g', label="Control Points")
    #if kbezier>1:
    plt.quiver(Pbeziertorch[0,::skipn,0].numpy(),Pbeziertorch[0,::skipn,1].numpy(),Pdot_t[0,::skipn,0].numpy(),Pdot_t[0,::skipn,1].numpy(), angles='xy', color='black', label="Velocity Vectors")
    

    plt.legend(loc='best')
    # plt.savefig("/home/ttw2xk/f1_figures/bezier_examples/example_%d.png" %kbezier)
    # plt.savefig("/home/ttw2xk/f1_figures/bezier_examples/example_%d.svg" %kbezier)
    # plt.savefig("/home/ttw2xk/f1_figures/bezier_examples/example_%d.eps" %kbezier)
plt.legend(loc="best")
plt.show()