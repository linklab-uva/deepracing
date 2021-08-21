import torch
import torch.distributions as D
import torch.nn as NN, torch.nn.functional as F, torchvision.transforms.functional as VF
import deepracing_models.math_utils as mu, deepracing_models
from deepracing_models.nn_models.VariationalModels import VariationalCurvePredictor
from matplotlib import pyplot as plt
import time
import numpy as np
import PIL, PIL.Image as PILImage
import yaml
import matplotlib.animation as animation
from typing import List
import random

batch_size = 1
d = 2

imrows = 66
imcols = 200
output_dimension = 2
steps = 240
gpu = 0
d = 2
s = torch.linspace(0.0, 1.0, steps=steps, dtype=torch.float64, device=torch.device("cuda:%d" %(gpu,))).unsqueeze(0).repeat(batch_size,1)

# print(state_dict)
# model_dir = "D:/f1_model_files/NeurIPS_demo/bezier/australia/12806a16bde94fa6ac40fb7a7cd13f09/%s"
model_dir = "D:/f1_model_files/variationalcurvepredictor/e160a2b5d43047e69796d1b2719b32c1/%s"
with open(model_dir % "epoch_200_params.pt","rb") as f:
    state_dict = torch.load(f, map_location=s.device)
with open(model_dir % "model_config.yaml","r") as f:
    config = yaml.load(f,Loader=yaml.SafeLoader)
print(config)
bezier_order = config["bezier_order"]
fix_first_point = config["fix_first_point"]
hidden_dimension = config["hidden_dimension"]
context_length = config["context_length"]
imchannels = config["input_channels"]
M = mu.bezierM(s, bezier_order)
Msquare = torch.square(M)
network = VariationalCurvePredictor(bezier_order = bezier_order, fix_first_point=fix_first_point, output_dimension=d, input_channels = imchannels, context_length = context_length, hidden_dim=hidden_dimension).type(s.dtype).to(s.device)
network.load_state_dict(state_dict, strict=False)
network = network.train()
cfgnet = vars(network)
cfgout = {k:v for (k,v) in cfgnet.items() if type(v) in {int,float,str,bool,List[int],List[float],List[str],List[bool]} }
print(cfgout)

imdir = "D:/f1_training_data/trent_solo_2/images/image_%d.jpg"
input_images = torch.empty(context_length, imchannels, imrows, imcols, dtype=s.dtype, device=s.device)
display_images = []
imstart = random.randint(200,2000)
#imstart = 321
imrange = range(imstart,imstart+context_length)
print("Testing on images: " + str(["image_%d.jpg" % i for i in imrange]))
for i in imrange:
    impil = PILImage.open(imdir%i)
    imtensor = VF.resized_crop(impil, 32, 0, 360, 1758, (imrows,imcols), interpolation=PILImage.LANCZOS)
    idx = i-imstart
    input_images[idx] = VF.to_tensor(imtensor).type(s.dtype).to(s.device)
    display_images.append(impil)
input_images = input_images.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
images_np = np.array([np.array(im) for im in display_images])[:,32:]
print(images_np.shape)

means, varfactors, covarfactors = network(input_images)
meanseval = torch.matmul(M, means)
print(meanseval.shape)
# if fix_first_point:
#     means = torch.cat([torch.zeros_like(means_[:,0]).unsqueeze(1), means_], dim=1)
#     varfactors = torch.cat([1E-7*torch.ones_like(varfactors_[:,0]).unsqueeze(1), varfactors_], dim=1)
#     covarfactors = torch.cat([torch.zeros_like(covarfactors_[:,0]).unsqueeze(1), covarfactors_], dim=1)
# else:
#     means = means_
#     varfactors = varfactors_
#     covarfactors = covarfactors_
scale_trils = torch.diag_embed(varfactors) + torch.diag_embed(covarfactors, offset=-1)
print("scale_trils.shape: " + str(scale_trils.shape))

distcurves = D.MultivariateNormal(means, scale_tril=scale_trils)
curvesamples = distcurves.sample((64,))[:,0]
print(curvesamples.shape)
sampleseval = torch.matmul(M[0], curvesamples).detach().cpu().numpy()
print(sampleseval.shape)

covars = torch.matmul(scale_trils, scale_trils.transpose(2,3))
covars_expand = covars.unsqueeze(1).expand(batch_size, Msquare.shape[1], Msquare.shape[2], d, d)
covarstacks = covars_expand*Msquare[:,:,:,None,None]
covarpoints = torch.sum(covarstacks, dim=2)
meanpoints = torch.matmul(M, means)
distpoints = torch.distributions.MultivariateNormal(meanpoints, covariance_matrix=covarpoints, validate_args=False)
covars = covarpoints[0,:,1,0]
sigmax = torch.sqrt(covarpoints[0,:,0,0])
sigmay = torch.sqrt(covarpoints[0,:,1,1])
ux = meanseval[0,:,0]
uy = meanseval[0,:,1]
correlationcoefficients = (covars/(sigmax*sigmay))
print("correlationcoefficients.shape: " + str(correlationcoefficients.shape))
conditionalvars = (1.0 - torch.square(correlationcoefficients))*(covarpoints[0,:,0,0])
conditionaldists = D.Normal(ux.unsqueeze(1), torch.sqrt(conditionalvars).unsqueeze(1))
uynp = uy.detach().cpu().numpy()
leftvals = conditionaldists.icdf(torch.ones(1, dtype=s.dtype, device=s.device)*.01)[:,0].detach().cpu().numpy()
print("leftvals.shape: " + str(leftvals.shape))
rightvals = conditionaldists.icdf(torch.ones(1, dtype=s.dtype, device=s.device)*.99)[:,0].detach().cpu().numpy()

labels = meanpoints.clone().detach()# + 0.01*torch.randn_like(meanpoints)

logprobs = distpoints.log_prob(labels)
probs = torch.exp(logprobs)

loss = torch.mean(-logprobs)
loss.backward()
#

ptssamp = distpoints.sample((256,))

a = ptssamp[:,0,::int(steps/8)]
b = ptssamp[:,0,-1].unsqueeze(1)
gaussianpoints = torch.cat([a , b], dim=1).detach().cpu().numpy()
acovar = covarpoints[0,::int(steps/8)]
bcovar = covarpoints[0,-1].unsqueeze(0)
print("acovar.shape: " + str(acovar.shape))
print("bcovar.shape: " + str(bcovar.shape))
gaussiancovars = torch.cat([acovar, bcovar], dim=0)
print(gaussiancovars)


ameanpoints = meanpoints[0,::int(steps/8)]
bmeanpoints = meanpoints[0,-1].unsqueeze(0)
mpsamp = torch.cat([ameanpoints, bmeanpoints], dim=0).detach().cpu().numpy()
print("mpsamp.shape: " + str(mpsamp.shape))


eigenvals, eigenvecs = torch.symeig(gaussiancovars, eigenvectors=True)
dominanteigenvals = eigenvals[:,-1].detach().cpu().numpy()
dominanteigenvecs = eigenvecs[:,:,1].detach().cpu().numpy()
print(dominanteigenvals)
print(dominanteigenvecs)



ptsnp = meanpoints[0].detach().cpu().numpy()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False)

#images_np = np.array([np.array(VF.to_pil_image(input_images[0,i])) for i in range(context_length)])

#image_np_transpose=skimage.util.img_as_ubyte(images_np[-1].transpose(1,2,0))
# oap = other_agent_positions[other_agent_positions==other_agent_positions].view(1,-1,60,2)
# print(oap)
ims = []
for i in range(images_np.shape[0]):
    ims.append([ax1.imshow(images_np[i])])
ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True, repeat=True)
meanx = ptsnp[:,0]
meany = ptsnp[:,1]
allx = gaussianpoints[:,:,0]
ally = gaussianpoints[:,:,1]
uynp = uy.detach().cpu().numpy()
ax2.set_xlim(np.max(allx)+1.0,np.min(allx)-1.0)
for i in range(gaussianpoints.shape[1]):
    ax2.scatter(gaussianpoints[:,i,0],gaussianpoints[:,i,1])
    ax2.quiver(mpsamp[:,0], mpsamp[:,1], dominanteigenvecs[:,0], dominanteigenvecs[:,1])
for i in range(sampleseval.shape[0]):
    ax2.plot(sampleseval[i,:,0], sampleseval[i,:,1], linestyle='dashed')
ax2.plot(leftvals,uynp,c="r")
ax2.plot(rightvals,uynp,c="r")
ax2.plot(meanx,meany,c="b")


ax3.plot(meanx,meany,c="b")
ax3.set_xlim(np.max(allx)+1.0,np.min(allx)-1.0)
for i in range(curvesamples.shape[1]):
    ax3.scatter(curvesamples[:,i,0].detach().cpu().numpy(), curvesamples[:,i,1].detach().cpu().numpy())


plt.show()
