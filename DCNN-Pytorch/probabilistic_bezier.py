import torch
import torch.distributions as D
import torch.nn as NN, torch.nn.functional as F
import deepracing_models.math_utils as mu
from matplotlib import pyplot as plt
import time
import numpy as np

batch_size = 64
kbezier = 5
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






varfactors = torch.zeros(batch_size, numcontrolpoints, d, device=device, dtype=dtype, requires_grad=True) + 1E-3
# varfactors[:,0] = 1E-3
varfactors[:,:,0] = 1.0
varfactors[:,:,1] = 1.0
# varfactors.requires_grad = True
#covarfactors = 0.1*torch.randn(batch_size, numcontrolpoints, 1, device=device, dtype=dtype, requires_grad=True)
covarfactors = torch.zeros(batch_size, numcontrolpoints, 1, device=device, dtype=dtype, requires_grad=True)
# covarfactors[:,0] = 0.0
# covarnoise = torch.randn_like(covarfactors[:,1:])
# covarfactors.requires_grad = True
#co_stdevs = 
# scale_trils = 0.1*torch.randn(batch_size, numcontrolpoints, d, d, device=device, dtype=dtype, requires_grad=True)
# scale_trils[:,:,0,0] = torch.abs(scale_trils[:,:,0,0]) + 1E-3
# scale_trils[:,:,1,1] = torch.abs(scale_trils[:,:,1,1]) + 1E-3
# scale_trils[:,:,0,1] = 0.0
  #  scale_trils[:,i+1,i]+=0.25*np.random.randn()
# scale_trils.requires_grad = True
# for i in range(1, numcontrolpoints):
#     varfactors[:,i,0] = 0.1*i
#     varfactors[:,i,1] = 0.25*i
#     covarfactors[:,i] *= i
    #covarfactors[:,i] += torch.randn_like(covarfactors[:,i], requires_grad=True)

scale_trils = torch.diag_embed(varfactors) + torch.diag_embed(covarfactors, offset=-1)
print(scale_trils[0])
    



p0 = torch.zeros(batch_size, 2, device=device, dtype=dtype, requires_grad = True)
pf = p0.clone()
# pf[:,:,0]=25.0
pf[:,1]=25.0
delta = pf - p0

means = torch.stack([p0 + s.item()*delta for s in torch.linspace(0.0,1.0,steps=numcontrolpoints)], dim=1)
#means[:,1:]+=2.0*torch.randn_like(means[:,1:])
meansflat = means.view(batch_size,-1)

batchdist = D.MultivariateNormal(means, scale_tril=scale_trils, validate_args=True)


meansout = batchdist.mean.view(batch_size, numcontrolpoints, d)
meanseval = torch.matmul(M,meansout)
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
print(covarpositions[0,0:10])
print(s[0,0:10])
print(torch.sqrt(s[0,0:10]))
# print(covarpositionfactors[0,1:4])
# print(scaletrilpositions[0,1:4])

distpos = D.MultivariateNormal(meanseval, covariance_matrix=covarpositions, validate_args=False)
# distpos2 = D.MultivariateNormal(meanseval, scale_tril=scaletrilpositions, validate_args=False)

log_probs = distpos.log_prob(meanseval)
# minus_log_probs = -batchdist.log_prob(meansflat)
probs = torch.exp(log_probs)

batchidx = 0
print(log_probs.shape)
print(log_probs[batchidx])
print(probs[batchidx])
loss = torch.mean(-log_probs)
print(loss)
loss.backward()
print(loss)

N = 128
samplespos = distpos.sample((N,))
print(samplespos.shape)
samplesposnp = samplespos.detach().cpu().numpy()
fig1 = plt.figure()
# plt.scatter(samplesposnp[0,0,:,0], samplesposnp[0,0,:,1])
# for i in range(N):
#     plt.scatter(samplesposnp[i,0,:,0], samplesposnp[i,0,:,1])
for i in range(nsteps):
    plt.scatter(samplesposnp[:,batchidx,i,0], samplesposnp[:,batchidx,i,1])
plt.plot(meanseval[batchidx,:,0].detach().cpu().numpy(), meanseval[batchidx,:,1].detach().cpu().numpy(), c="black")
plt.xlim(-3,3)
# fig2 = plt.figure()
#plt.plot(probs[batchidx,1:].detach().cpu().numpy())
plt.show()

