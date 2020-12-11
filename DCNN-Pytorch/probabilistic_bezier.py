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






stdevs = 0.2*torch.ones(batch_size, numvars ,device=device, dtype=dtype) + 1E-3
#co_stdevs = 

scale_trils = torch.diag_embed(stdevs) #+ torch.diag()
#scale_trils[covarmask] = co_stdevs.flatten()

scale_trils[:,0,0] = scale_trils[:,1,1] = 1E-2# = scale_trils[:,-1,-1] = scale_trils[:,-2,-2] = 1E-7
scale_trils[:,1,0] = 0.0

# scale_trils_rs = scale_trils[control_point_mask].view(batch_size, numcontrolpoints, d,d)

for i in range(d, numvars, d):
    upfactor = int(i/d)
    scale_trils[:,i,i]*=upfactor
    scale_trils[:,i+1,i+1]*=5.0*upfactor
  #  scale_trils[:,i+1,i]+=0.25*np.random.randn()
print(scale_trils[0,0:6,0:6])
scale_trils.requires_grad = True
    



p0 = torch.zeros(batch_size, 1, 2, device=device, dtype=dtype)
pf = p0.clone()
# pf[:,:,0]=25.0
pf[:,:,1]=25.0
delta = pf - p0

means = torch.cat([p0 + s.item()*delta for s in torch.linspace(0.0,1.0,steps=numcontrolpoints)], dim=1)
means.requires_grad = True
#means[:,1:]+=2.0*torch.randn_like(means[:,1:])
meansflat = means.view(batch_size,-1)

# batchstandard = D.MultivariateNormal(torch.zeros(batch_size, numvars, dtype=dtype, device=device), scale_tril=*torch.eye(numvars, dtype=dtype, device=device).unsqueeze(0).repeat(batch_size,1,1),validate_args=True)
batchstandard = D.MultivariateNormal(meansflat, scale_tril=torch.eye(numvars, dtype=dtype, device=device).unsqueeze(0).repeat(batch_size,1,1),validate_args=True)
batchdist = D.MultivariateNormal(meansflat, scale_tril=scale_trils,validate_args=True)


meansout = batchdist.mean.view(batch_size, numcontrolpoints, d)
meanseval = torch.matmul(M,meansout)

Sigma =  batchdist.covariance_matrix
Sigma_rs = batchdist.covariance_matrix[control_point_mask].view(batch_size, numcontrolpoints, d,d)


Msquare = torch.square(M)

# print(Msquare.shape)
Sigma_unsqueeze = Sigma_rs.unsqueeze(1).expand(batch_size,Msquare.shape[1], numcontrolpoints, d, d)
# print(scale_trils_unsqueeze.shape)
# example = (Msquare[:,0])[:,None]*Sigma_rs
# print(example.shape)
# covarstacks = Sigma_unsqueeze*Msquare[:,:,:,None,None]
covarstacks = Sigma_unsqueeze*Msquare[:,:,:,None,None]
covarpositions = torch.sum(covarstacks, dim=2)
print(covarpositions.shape)

# row = 50
# trilmanual = torch.zeros(batch_size, Msquare.shape[1], d, d, dtype=dtype, device=device)
# covarmanual = torch.zeros(batch_size, Msquare.shape[1], d, d, dtype=dtype, device=device)
# # print(covarmanual.shape)
# for b in range(batch_size):
#     for i in range(covarmanual.shape[1]):
#         factors = Msquare[b,i]
#         for j in range(factors.shape[0]):
#             trilmanual[b,i]+=factors[j]*scale_trils_rs[b,j]
#             covarmanual[b,i]+=factors[j]*Sigma_rs[b,j]

# print(covarpositions[0,row])
# print(covarmanual[0,row])
# print(trilpositions[0,row])
# print(trilmanual[0,row])
# distpos = D.MultivariateNormal(sampleseval, scale_tril=trilpositions, validate_args=True)
distpos = D.MultivariateNormal(meanseval, covariance_matrix=covarpositions, validate_args=False)

minus_log_probs = distpos.log_prob(meanseval)
# minus_log_probs = -batchdist.log_prob(meansflat)
probs = torch.exp(minus_log_probs)

batchidx = 0
print(minus_log_probs.shape)
print(probs[batchidx])
loss = torch.mean(-minus_log_probs)
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

