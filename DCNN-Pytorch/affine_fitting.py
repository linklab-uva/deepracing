import numpy.linalg as la
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
import matplotlib
from matplotlib import pyplot as plt
import deepracing_models.math_utils as mu

N=500
minimum_singular_value = 0.0
xin = torch.linspace(0.0,5.0,steps=N, dtype=torch.float32) 
yin = xin**2

Xin = torch.stack([xin,yin,torch.ones_like(xin)],dim=1)
Xinnoisy = Xin.clone()
Xinnoisy[:,1]+= 0.25*torch.randn_like(Xinnoisy[:,1])
#Xinnoisy[2,0]=Xinnoisy[3,0]=Xinnoisy[4,0]
# Ttrue = torch.eye(3, dtype=Xin.dtype, device=Xin.device)
# Ttrue[0:2] = 0.2*torch.randn_like(Ttrue[0:2])
Ttrue = torch.randn(3,2)

Xout = torch.matmul(Xin, Ttrue)
Xoutnoisy = Xout.clone()
Xoutnoisy[:,1]+=0.25*torch.randn_like(Xoutnoisy[:,1])


# U,S,V = torch.svd(Xin)
# sinv =  torch.where(S > minimum_singular_value, 1/S, torch.zeros_like(S))
# pinv = torch.matmul(torch.matmul(V,torch.diag_embed(sinv).t()),U.t())
# lsqres = torch.matmul(pinv, Xoutnoisy)
# lsqres, Qr = torch.lstsq(Xoutnoisy, Xinnoisy)
# Tfit = torch.cat([lsqres[0:3], torch.zeros_like(lsqres[0:3,0]).unsqueeze(1)], dim=1).t()
# Tfit[-1,-1]=1.0
#Tfit = torch.inverse(Tfit)
#print(Tfit.shape)

# Xin_, Tfit = mu.fitAffine(Xin[:,0:2], Xoutnoisy)
Xin_, Tfit = mu.fitAffine(Xinnoisy[:,0:2], Xoutnoisy)

# Xoutfit = torch.matmul(Xin, Tfit)
Xoutfit = torch.matmul(Xinnoisy, Tfit)

print(Ttrue)
print(Tfit)



fig = plt.figure()

plt.plot(Xin[:,0].cpu().numpy(), Xin[:,1].cpu().numpy(), label="Input data")
plt.plot(Xinnoisy[:,0].cpu().numpy(), Xinnoisy[:,1].cpu().numpy(), label="Noisy Input data")
plt.plot(Xout[:,0].cpu().numpy(), Xout[:,1].cpu().numpy(), label="Transformed data")
plt.scatter(Xoutnoisy[:,0].cpu().numpy(), Xoutnoisy[:,1].cpu().numpy(), label="Noisy transformed data")
plt.scatter(Xoutfit[:,0].cpu().numpy(), Xoutfit[:,1].cpu().numpy(), label="Least Squares Fit")
plt.legend()

plt.show()


