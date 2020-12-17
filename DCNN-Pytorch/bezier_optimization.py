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
import torch, torch.nn as nn
import deepracing_models.math_utils as mu
import deepracing_models.nn_models.LossFunctions as LF
import time
from deepracing_models.math_utils import BezierCurveModule


dev = torch.device("cuda:1")
# dev = torch.device("cpu")
bezier_order = 7
steps = 150

s = torch.linspace(0.0, 1.0, steps=steps, dtype=torch.float32, device=dev).unsqueeze(0)
s.requires_grad=False

bezier_order_boundary = 5
sboundary = torch.linspace(0.0, 1.0, steps=20000, dtype=s.dtype, device=s.device).unsqueeze(0).repeat(2,1)
Mboundary = mu.bezierM(sboundary, bezier_order_boundary)


p0 = torch.tensor([-3.0, -1.0], dtype=s.dtype, device=s.device)
pf = torch.tensor([-3.0, 110.0], dtype=s.dtype, device=s.device)
delta = pf - p0

s_control_points = torch.linspace(0.0, 1.0, steps=bezier_order_boundary+1, dtype=s.dtype, device=s.device)
boundary_curves = torch.zeros(2, bezier_order_boundary+1, 2, dtype=s.dtype, device=s.device)
boundary_curves[0] = p0 + delta*(s_control_points[:,None])
boundary_curves[1] = boundary_curves[0].clone()
boundary_curves[1,:,0]+=6.0

boundary_noise = 15.0*torch.randn_like(boundary_curves[0,3:-1])
boundary_curves[0,3:-1]+=boundary_noise
boundary_curves[1,3:-1]+=boundary_noise


boundaries = torch.matmul(Mboundary, boundary_curves)

_, boundarytangents = mu.bezierDerivative(boundary_curves, t=sboundary) 
boundarytangentnorms = torch.norm(boundarytangents, p=2, dim=2) + 1E-4
boundarytangents = boundarytangents/boundarytangentnorms[:,:,None]

boundarynormals =  boundarytangents.flip([2])
boundarynormals[:,:,0]*=-1.0
boundarynormals[1]*=-1.0


numcurves = 1

inner_boundary = boundaries[1].unsqueeze(0).repeat(numcurves,1,1)
inner_boundary_tangent = boundarytangents[1].unsqueeze(0).repeat(numcurves,1,1)
inner_boundary_normal = boundarynormals[1].unsqueeze(0).repeat(numcurves,1,1)
outer_boundary = boundaries[0].unsqueeze(0).repeat(numcurves,1,1)
outer_boundary_tangent = boundarytangents[0].unsqueeze(0).repeat(numcurves,1,1)
outer_boundary_normal = boundarynormals[0].unsqueeze(0).repeat(numcurves,1,1)




# theta = torch.linspace(3.0*np.pi/2, np.pi/2, steps=steps, dtype=torch.float64, device=dev)
# r = 5.0*torch.ones_like(theta)
# x = r*torch.cos(theta)
# y = 5.0 + r*torch.sin(theta)
# ig_points = torch.stack([x,y],dim=1).unsqueeze(0)
# print(ig_points.shape)

# M, initial_guess = mu.bezierLsqfit(ig_points, bezier_order, t=s)
# initial_guess = initial_guess[0]

initial_guess = (torch.tensor([0.0, 0.0], device=s.device, dtype=s.dtype) + torch.linspace(0,1,steps=bezier_order+1, dtype=s.dtype, device=s.device)[:,None]*torch.tensor([0.0, 100.0], device=s.device, dtype=s.dtype)).unsqueeze(0) 
initial_guess = initial_guess.repeat(numcurves,1,1)
mask = [True for asdf in range(bezier_order+1)]
mask[0] = False
ignoise = 1.0*torch.randn_like(initial_guess[:,mask])
#initial_guess[:,mask]+=ignoise
# initial_guess = 10.0*torch.tensor([[[ 0.0,  0.0],
#                                 [-3.0623e-01,  5.4987e-01],
#                                # [-9.1757e-01,  1.1755e+00],
#                                 [-5.3475e-01,  3.5227e+00],
#                                 [-5.3475e-01,  6.4773e+00],
#                               #  [-9.1757e-01,  8.245e+00],
#                                 [-3.5623e-01,  9.0],
#                                  [ -2.0e-01,  10.0]]], device=s.device, dtype=s.dtype) 
                               # [ 0.0,  10.0]]], device=s.device, dtype=s.dtype)

M = mu.bezierM(s,bezier_order)
Mder = mu.bezierM(s, bezier_order-1)
M2ndder = mu.bezierM(s, bezier_order-2)


#mask = None
print("Making model")
bcmodel = BezierCurveModule(initial_guess.clone(), mask=mask).type(s.dtype).to(s.device)
# print(bcmodel)
# print(bcmodel.state_dict())

# mse = nn.MSELoss().type(p_i.dtype).to(dev)
mse = LF.SquaredLpNormLoss()
#obstacleloss = LF.OtherAgentDistanceLoss(alpha = 10.0, beta=0.1)
# ibloss = LF.BoundaryLoss(inner_boundary, inner_boundary_normal, p=2, relu_type="")
ibloss = LF.BoundaryLoss(p=2, relu_type="Elu", alpha=3.0, beta=0.5, time_reduction="max").type(s.dtype).to(s.device)
obloss = LF.BoundaryLoss(p=2, relu_type="Elu", alpha=3.0, beta=0.5, time_reduction="max").type(s.dtype).to(s.device)
obstacle = torch.zeros(1,19, s.shape[1], 2, dtype=s.dtype, device=s.device)
obstacle [:,0,:,0] = -.9
obstacle [:,0,:,1] = torch.linspace(3.0, 7.0, s.shape[1], dtype=s.dtype, device=s.device )
valid = torch.zeros(1,19, dtype=torch.int64)
valid[:,0]=1
valid = valid.bool()
obstacleloss = LF.OtherAgentDistanceLoss(alpha=2.0).type(s.dtype).to(s.device)
mseloss = nn.MSELoss().type(s.dtype).to(s.device)

#optimizer = torch.optim.Adam(bcmodel.parameters(), lr=0.2)
#optimizer = torch.optim.RMSprop(bcmodel.parameters(), lr=1.0)

dT = 1.5
maxacent = 9.8*3
maxalinear = 7.0
nstep = 5
outputhistory = []
ibloss_ = torch.tensor([1.0])[0]
obloss_ = torch.tensor([1.0])[0]
stepcount = 0
tick = time.time()
optimizer = torch.optim.SGD(bcmodel.parameters(), lr=0.5, momentum=0.1)
while (True or ibloss_.item()>0.0 or obloss_.item()>0.0) and stepcount<nstep:
    outputs = bcmodel(M)
    stepfactor = np.sqrt((nstep-stepcount)/nstep)
    # stepfactor = 1.0
#     obstaclefactor = stepcount/nstep
#     obstaclefactor = 0.0
    all_control_points = bcmodel.allControlPoints()
    _, v_s = mu.bezierDerivative(all_control_points, M=Mder)
    v_t = v_s/dT
    speeds = torch.norm(v_t, p=2, dim=2)
    curvetangents = v_t/speeds[:,:,None]
    speedsquares = torch.square(speeds)
    speedcubes = speedsquares*speeds

    _, a_s =  mu.bezierDerivative(all_control_points, M=M2ndder, order=2)
    a_t = a_s/(dT*dT)
    accels = torch.norm(a_t, p=2, dim=2)
    v_text = torch.cat([v_t, torch.zeros_like(v_t[:,:,0]).unsqueeze(2)], dim=2)
    a_text = torch.cat([a_t, torch.zeros_like(a_t[:,:,0]).unsqueeze(2)], dim=2)
    radii = (speedcubes/(torch.norm(torch.cross(v_text,a_text, dim=2), p=2, dim=2) + 1E-3))# + 1.0
    angvels = speeds/radii
    centriptelaccels = speedsquares/radii

    ib_idx, ibloss_ = ibloss(outputs, inner_boundary, inner_boundary_normal) 
    ob_idx, obloss_ = obloss(outputs, outer_boundary, outer_boundary_normal)

    loss = stepfactor*ibloss_ + stepfactor*obloss_   + 0.0*stepfactor*torch.max(torch.nn.functional.relu(accels-maxalinear)) - 0.5*stepfactor*torch.mean(speeds) + 0.5*stepfactor*torch.max(torch.nn.functional.relu(centriptelaccels-maxacent))
     
    # loss = stepfactor*ibloss_ + stepfactor*obloss_ + 0.1*torch.max(centriptelaccels)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    stepcount+=1
 #   outputhistory.append(bcmodel.allControlPoints())
tock = time.time()
print("optimization took %d steps and %f seconds." %(stepcount, tock-tick))
print(bcmodel.allControlPoints()[0])
# outputhistory = torch.cat([pts for pts in outputhistory], dim=0)
# print(outputhistory.shape)
with torch.no_grad():
    outputs_np = bcmodel(M=M)[0].cpu().numpy()
    initial_guess_np = torch.matmul(M, initial_guess)[0].cpu().numpy()
    inner_boundary_np = inner_boundary[0].cpu().numpy()
    outer_boundary_np = outer_boundary[0].cpu().numpy()
    inner_boundary_normals_np = inner_boundary_normal[0].cpu().numpy()
    outer_boundary_normals_np = outer_boundary_normal[0].cpu().numpy()
    # outputhistoryeval = torch.matmul(M[0], outputhistory).cpu().numpy()
numarrows = int(outer_boundary_np.shape[0]/20)

anglestorch = (torch.atan2(outer_boundary_normal[:,:,1], outer_boundary_normal[:,:,0]) + 2.0*np.pi) % (2.0*np.pi)
anglesnp = ((180.0*anglestorch[0].cpu().numpy())/np.pi)[::numarrows]


fig = plt.figure()
# targets_np = targets[0].cpu().detach().numpy()
# plt.scatter(targets_np[:,0], targets_np[:,1], label="Targets",  marker='o', facecolors='none', edgecolors="g")
plt.plot(initial_guess_np[:,0], initial_guess_np[:,1], c="b", label="Initial Guess Bezier Curve")
plt.plot(outputs_np[:,0], outputs_np[:,1], c="r", label="Final Optimized Bezier Curve")
plt.plot(inner_boundary_np[:,0], inner_boundary_np[:,1], label="Inner Boundary")
plt.plot(outer_boundary_np[:,0], outer_boundary_np[:,1], label="Outer Boundary")
#plt.quiver(outer_boundary_np[::numarrows,0], outer_boundary_np[::numarrows,1], outer_boundary_normals_np[::numarrows,0], outer_boundary_normals_np[::numarrows,1], angles = "xy", units ="inches", scale_units ="inches", scale=2.0 )
# for i in range(outputhistoryeval.shape[0]-1):
#     plt.plot(outputhistoryeval[i,:,0], outputhistoryeval[i,:,1],'--', label="Optimization step %d" %(i+1,))
plt.legend()
plt.xlim(-10.0,10.0)
plt.ylim(-4.0,120.0)
plt.show()

