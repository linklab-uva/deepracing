import torch
import torch.distributions as D
import torch.nn as NN, torch.nn.functional as F
import deepracing_models.math_utils as mu
from matplotlib import pyplot as plt
import time
import numpy as np

batch_size = 64
kbezier = 3
numcontrolpoints = kbezier + 1 
d = 2
numvars = numcontrolpoints*d
numcovars = numcontrolpoints*(d-1)
device = torch.device("cuda:0")
dtype = torch.float32
nsteps = 8
s = torch.linspace(0,1.0,steps=nsteps,device=device,dtype=dtype).unsqueeze(0).repeat(batch_size,1)
M = mu.bezierM(s, kbezier)

tick = time.time()
covarmask = torch.zeros(batch_size, numvars, numvars, dtype=torch.int64)
control_point_mask = torch.eye(numvars, dtype=torch.int64).unsqueeze(0).repeat(batch_size,1,1)
for i in range(numcovars):
    covarmask[:,2*i+1,2*i] = 1
    control_point_mask[:,2*i+1,2*i] = 1
    control_point_mask[:,2*i,2*i+1] = 1
covarmask = covarmask.bool()
control_point_mask = control_point_mask.bool()
tock = time.time()






varfactors = torch.stack([torch.linspace(1E-3,0.25,steps=numcontrolpoints, device=device, dtype=dtype, requires_grad=True), torch.linspace(1E-3,1.5,steps=numcontrolpoints, device=device, dtype=dtype, requires_grad=True)], dim=1).unsqueeze(0).expand(batch_size,numcontrolpoints,d)
covarfactors = 0.001*torch.randn(1, numcontrolpoints, int(d*(d-1)/2), device=device, dtype=dtype, requires_grad=True).expand(batch_size,numcontrolpoints,int(d*(d-1)/2))
scale_trils = torch.diag_embed(varfactors) + torch.diag_embed(covarfactors, offset=-1)
print("scale_trils.shape: " + str(scale_trils.shape))
print(scale_trils[0])


p0 = torch.zeros(batch_size, 2, device=device, dtype=dtype, requires_grad = True)
pf = p0.clone()
pf[:,1]=30.0
delta = pf - p0

means = torch.stack([p0 + s.item()*delta for s in torch.linspace(0.0,1.0,steps=numcontrolpoints)], dim=1)
# means[:,1:-1]+=0.25*torch.randn_like(means[:,1:-1])

batchdist = D.MultivariateNormal(means, scale_tril=scale_trils, validate_args=True)


meansout = batchdist.mean.view(batch_size, numcontrolpoints, d)
meanseval = torch.matmul(M,meansout)

Mvel, meanvels = mu.bezierDerivative(means, t=s, order=1)

Msquare = torch.square(M)

# print(Msquare.shape)
# Sigma_unsqueeze = Sigma_rs.unsqueeze(1).expand(batch_size,Msquare.shape[1], numcontrolpoints, d, d)
sigma_control_points = batchdist.covariance_matrix
# print(sigma_control_points.shape)
scale_trils_unsqueeze = scale_trils.unsqueeze(1).expand(batch_size,Msquare.shape[1], numcontrolpoints, d, d)
sigma_control_points_unsqueeze = sigma_control_points.unsqueeze(1).expand(batch_size,Msquare.shape[1], numcontrolpoints, d, d)
# print(scale_trils_unsqueeze.shape)
# example = (Msquare[:,0])[:,None]*Sigma_rs
# print(example.shape)
# covarstacks = Sigma_unsqueeze*Msquare[:,:,:,None,None]
scale_tril_stacks = scale_trils_unsqueeze*M[:,:,:,None,None]
scaletrilpositions = torch.sum(scale_tril_stacks, dim=2)
covarstacks = sigma_control_points_unsqueeze*Msquare[:,:,:,None,None]
covarpositions = torch.sum(covarstacks, dim=2)
covarpositionfactors = torch.cholesky(covarpositions)



distpos = D.MultivariateNormal(meanseval, covariance_matrix=covarpositions, validate_args=False)

covars = covarpositions[0,:,1,0]
sigmax = torch.sqrt(covarpositions[0,:,0,0])
sigmay = torch.sqrt(covarpositions[0,:,1,1])
ux = meanseval[0,:,0]
uy = meanseval[0,:,1]
correlationcoefficients = (covars/(sigmax*sigmay))
print("correlationcoefficients.shape: " + str(correlationcoefficients.shape))
conditionalvars = (1.0 - torch.square(correlationcoefficients))*(covarpositions[0,:,0,0])

conditionaldists = D.Normal(ux.unsqueeze(1), torch.sqrt(conditionalvars).unsqueeze(1))

uynp = uy.detach().cpu().numpy()
leftvals = conditionaldists.icdf(torch.ones(1, dtype=s.dtype, device=s.device)*.02)[:,0].detach().cpu().numpy()
rightvals = conditionaldists.icdf(torch.ones(1, dtype=s.dtype, device=s.device)*.98)[:,0].detach().cpu().numpy()
# rightvals = -leftvals


log_probs = distpos.log_prob(meanseval)
probs = torch.exp(log_probs)

loss = torch.mean(-log_probs)
loss.backward()

N = 500

batchidx = 0
samplespos = distpos.sample((N,))
samplesposnp = samplespos.detach().cpu().numpy()
fig1 = plt.figure()
# plt.scatter(samplesposnp[0,0,:,0], samplesposnp[0,0,:,1])
# for i in range(N):
#     plt.scatter(samplesposnp[i,0,:,0], samplesposnp[i,0,:,1])
for i in range(nsteps):
    plt.scatter(samplesposnp[:,batchidx,i,0], samplesposnp[:,batchidx,i,1])
plt.plot(meanseval[batchidx,:,0].detach().cpu().numpy(), meanseval[batchidx,:,1].detach().cpu().numpy(), c="black")
plt.plot(leftvals, uynp, c="r")
plt.plot(rightvals, uynp, c="r")
plt.xlim(-3,3)
# fig2 = plt.figure()
#plt.plot(probs[batchidx,1:].detach().cpu().numpy())
plt.show()

