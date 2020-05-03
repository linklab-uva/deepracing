import torch
import deepracing_models.nn_models.Models as M
import time
net = M.AdmiralNetKinematicPredictor(use_3dconv=False, sequence_length=20, context_length=5)
net = net.cuda(0)
im = torch.rand(128,5,3,66,200)
im = im.cuda(0)
net=net.eval()
print(net)
print("Running net")
tick = time.time()
out = net(im)
tock = time.time()
print(out.shape)
print("Got prediction in %f seconds"%(tock-tick))