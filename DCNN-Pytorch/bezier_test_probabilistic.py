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
import torch.distributions
from torch.distributions import MultivariateNormal as MVN
import deepracing_models.distributions as customdist
def lsqfit(points,M):
    M_ = linalg.pinv(M)
    return M_ * points
pi = np.pi
num_points = 60
tmin = -1.0
tmax = 1.0
d = 7
kbezier = d
numcurves = 10
Mtk = lambda i, n, t: t**(i)*(1-t)**(n-i)*nOk(n,i)
bezierFactors = lambda ts, n: matrix([[Mtk(k_,n,t) for k_ in range(n+1)] for t in ts])
P = np.zeros((numcurves,num_points,2))
Pprime = np.zeros((numcurves,num_points,2))
Pprimeprime = np.zeros((numcurves,num_points,2))
tnp = np.zeros((numcurves,num_points))
t = np.linspace(tmin, tmax, num_points )
for i in range(0,numcurves):
    tnp[i] = t.copy()
    px = np.random.randn(d+1)
    py = np.random.randn(d+1)
    P[i,:,0] = np.polyval(px,tnp[i])
    P[i,:,1] = np.polyval(py,tnp[i])
    P[i,:,0] = P[i,:,0] - P[i,0,0] 
    P[i,:,1] = P[i,:,1] - P[i,0,1] 
    Pprime[i,:,0] = np.polyval(np.polyder(px, m=1),tnp[i])
    Pprime[i,:,1] = np.polyval(np.polyder(py, m=1),tnp[i])
    Pprimeprime[i,:,0] = np.polyval(np.polyder(px, m=2),tnp[i])
    Pprimeprime[i,:,1] = np.polyval(np.polyder(py, m=2),tnp[i])

Ptorch = torch.from_numpy(P.copy()).double().cuda(0)#.unsqueeze(0)
Pprimetorch = torch.from_numpy(Pprime.copy()).double().cuda(0)#.unsqueeze(0)
Pprimeprimetorch = torch.from_numpy(Pprimeprime.copy()).double().cuda(0)#.unsqueeze(0)
ttorch = torch.from_numpy(tnp.copy()).double().cuda(0)#.unsqueeze(0)
dt = ttorch[:,-1]-ttorch[:,0]
storch = (ttorch - ttorch[:,0,None])/dt[:,None]
numdims = 2*kbezier#+2
numsamples = 40
numdiagvars = numdims
numcovars = int((numdims*(numdims-1))/2)

M , bezier_control_points = math_utils.bezierLsqfit(Ptorch, kbezier, t = storch )

Mderiv, Pdot_s = math_utils.bezierDerivative(bezier_control_points,storch, order=1)
Pdot_t= Pdot_s/dt[:,None,None]

Mdotdot, Pdotdot_s = math_utils.bezierDerivative(bezier_control_points, storch, order=2)
Pdotdot_t= Pdotdot_s/((dt**2)[:,None,None])

print("diff in first point: %f" % (torch.max(torch.abs(bezier_control_points[:,0] - Ptorch[:,0]),)))
Pbeziertorch = torch.matmul(M,bezier_control_points)
first_points = bezier_control_points[:,0].clone().unsqueeze(1).repeat(1,numsamples,1).unsqueeze(2)
bezier_control_points = bezier_control_points[:,1:]
print("M.shape: %s" % (str(M.shape),))
print("bezier_control_points.shape: %s" % (str(bezier_control_points.shape),))
print("Pbeziertorch.shape: %s" % (str(Pbeziertorch.shape),))
bcflat = bezier_control_points.reshape(numcurves,-1) 
print("bcflat.shape: %s" % (str(bcflat.shape),))

scaleoff = 0.0125
scalediag = 0.25
signeddiagstdev = torch.randn((numcurves,numdiagvars),requires_grad=False).double().cuda()
diagsdtev = scalediag*torch.abs(signeddiagstdev)

signedoffdiagstdev = torch.randn((numcurves,numcovars),requires_grad=False).double().cuda()
offdiagstdev = scaleoff*signedoffdiagstdev
print("diagsdtev.shape: %s" % (str(diagsdtev.shape),))
scale_tril = torch.diag_embed(diagsdtev)
tril_indices = torch.tril_indices(row=numdims, col=numdims, offset=-1)
scale_tril[:, tril_indices[0], tril_indices[1]]=offdiagstdev
#print("scale_tril: %s" % (str(scale_tril),))
print("scale_tril.shape: %s" % (str(scale_tril.shape),))
print("scale_tril.requires_grad: %s" % (str(scale_tril.requires_grad),))


bcdist = MVN(bcflat,scale_tril=scale_tril,validate_args=True)
bcsamples_ = bcdist.sample((numsamples,))
print("bcsamples_.shape: %s" % (str(bcsamples_.shape),))
logprob = bcdist.log_prob(bcsamples_)
pdfvals = torch.exp(logprob).t()
#print("pdfvals: %s" % (str(pdfvals),))
pdfmaxes,pdfmaxes_idx = torch.max(pdfvals,dim=1,keepdim=True)
alphas = pdfvals/pdfmaxes
print("first_points.shape: %s" % (str(first_points.shape),))
#print("bcsamples_.shape: %s" % (str(bcsamples_.shape),))
bcsamples = torch.cat((first_points,bcsamples_.transpose(0,1).reshape(numcurves,numsamples,kbezier,2)),dim=2)
print("bcsamples.shape: %s" % (str(bcsamples.shape),))
#bcs
sample_curves = torch.matmul(M.unsqueeze(1),bcsamples)
print("sample_curves.shape: %s" % (str(sample_curves.shape),))


#print(scale_tril)

fig : Figure = plt.figure()
# print(curve_alphas_z)
for i in range(numcurves):
   #ax.plot(P[i,:,0],P[i,:,1],'r-')
    plt.title("A Probabilistic Bézier Curve (Mean in black).\n%d samples from the curve in blue, brightness weighted by probability density" % (numsamples,))
    for j in range(numsamples):
       # curve_alpha = float(curve_alphas_z[i,j].cpu().item())
        plt.scatter(sample_curves[i,j,:,0].cpu().numpy(),sample_curves[i,j,:,1].cpu().numpy(), facecolors='b', edgecolors='b', label="Sampled curve %d" %(j,), alpha=alphas[i,j].cpu().item())

   
    toplot = P[i]
    #plt.plot(toplot[:,0].cpu().numpy(),toplot[:,1].cpu().numpy(), color='red', label="Some points")
    plt.quiver(toplot[:,0], toplot[:,1], 0.25*Pprime[i,:,0], 0.25*Pprime[i,:,1], angles='xy', color='black')
    

    print("Mean distance: %f" %(torch.mean(torch.norm(Pbeziertorch[i]-Ptorch[i],dim=1,p=2)).item()))
    print("Mean velocity diff: %f" %(torch.mean(torch.norm(Pdot_t[i]-Pprimetorch[i],dim=1,p=2)).item()))
    print("Mean acceleration diff: %f" %(torch.mean(torch.norm(Pdotdot_t[i]-Pprimeprimetorch[i],dim=1,p=2)).item()))
    
    plt.show()