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

class BezierCurveModule(nn.Module):
    def __init__(self, control_points, move_first_point = False, move_last_point = False):
        super(BezierCurveModule, self).__init__()
      #  print(control_points.shape)
        self.control_points = nn.Parameter(control_points[:,1:-1], requires_grad=True)
       # print(self.control_points.shape)
        self.p_0 = nn.Parameter(control_points[:,0], requires_grad=move_first_point) 
        self.p_f = nn.Parameter(control_points[:,-1], requires_grad=move_last_point) 
       # self.allControlPoints_ = 
    def allControlPoints(self):
        return torch.cat([self.p_0.unsqueeze(1), self.control_points, self.p_f.unsqueeze(1)], dim=1)
    def forward(self, M):
        # if not ((s is not None) ^ (M is not None)):
        #     raise ValueError("Either s or M must be set, but not both")
        points = self.allControlPoints()
        return torch.matmul(M, points)

dev = torch.device("cuda:0")
bezier_order = 3
steps = 250

s = torch.linspace(0.0, 1.0, steps=steps, dtype=torch.float32, device=dev).unsqueeze(0)
s.requires_grad=False


inner_boundary = torch.zeros(20000,2, dtype=s.dtype, device=s.device)
inner_boundary[:,0] = -1.0
inner_boundary[:,1] = torch.linspace(-1,10, inner_boundary.shape[0], dtype=s.dtype, device=s.device)
inner_boundary.requires_grad=False

inner_boundary_normal=inner_boundary.clone()
inner_boundary_normal[:,1]=0.0
inner_boundary_normal.requires_grad=False


outer_boundary = inner_boundary.clone()
outer_boundary[:,0]=1.0
outer_boundary.requires_grad=False

outer_boundary_normal=outer_boundary.clone()
outer_boundary_normal[:,1]=0.0
outer_boundary_normal.requires_grad=False

inner_boundary = inner_boundary.unsqueeze(0)
inner_boundary_normal = inner_boundary_normal.unsqueeze(0)
outer_boundary = outer_boundary.unsqueeze(0)
outer_boundary_normal = outer_boundary_normal.unsqueeze(0)

# theta = torch.linspace(3.0*np.pi/2, np.pi/2, steps=steps, dtype=torch.float64, device=dev)
# r = 5.0*torch.ones_like(theta)
# x = r*torch.cos(theta)
# y = 5.0 + r*torch.sin(theta)
# ig_points = torch.stack([x,y],dim=1).unsqueeze(0)
# print(ig_points.shape)

# M, initial_guess = mu.bezierLsqfit(ig_points, bezier_order, t=s)
# initial_guess = initial_guess[0]

initial_guess = torch.tensor([[[ 0.0,  0.0],
                                [-2.0623e-01, -8.4987e-05],
                              #  [-9.1757e-01,  1.1755e+00],
                             #   [-2.3475e+00,  3.5227e+00],
                                [-4.3475e+00,  6.4773e+00],
                              #  [-9.1757e-01,  8.245e+00],
                          #      [-7.5623e-01,  9.0],
                                [ 0.0,  10.0]]], device=s.device, dtype=s.dtype) 
initial_guess[:,1:-1] += 0.3*torch.randn_like(initial_guess[:,1:-1])

M = mu.bezierM(s,bezier_order)

bcmodel = BezierCurveModule(initial_guess.clone()).type(s.dtype).to(dev)
# print(bcmodel)
# print(bcmodel.state_dict())

# mse = nn.MSELoss().type(p_i.dtype).to(dev)
mse = LF.SquaredLpNormLoss()
#obstacleloss = LF.OtherAgentDistanceLoss(alpha = 10.0, beta=0.1)
# ibloss = LF.BoundaryLoss(inner_boundary, inner_boundary_normal, p=2, relu_type="")
ibloss = LF.BoundaryLoss(p=2, relu_type="", alpha=1.0, beta=0.0, time_reduction="max")

optimizer = torch.optim.SGD(bcmodel.parameters(), lr=0.1, momentum=0.0)
#optimizer = torch.optim.Adam(bcmodel.parameters(), lr=0.1)
#optimizer = torch.optim.RMSprop(bcmodel.parameters(), lr=1.0)

nstep = 50
outputhistory = []
loss = torch.tensor([1.0])[0]
stepcount = 0
tick = time.time()
while loss.item()>0.0 and stepcount<nstep:
    outputs = bcmodel(M)
    loss = ibloss(outputs, inner_boundary, inner_boundary_normal )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    outputhistory.append(bcmodel.allControlPoints()[0].detach())
    stepcount+=1
outputhistory = torch.stack(outputhistory, dim=0).detach()
print(outputhistory.shape)
pointshistory = torch.matmul(M[0], outputhistory)
print(pointshistory.shape)
pointshistory_np = pointshistory.cpu().numpy()
tock = time.time()
print("optimization took %f seconds" %(tock-tick,))
with torch.no_grad():
    outputs_np = bcmodel(M=M)[0].cpu().numpy()
    print(bcmodel.allControlPoints())
    initial_guess_np = torch.matmul(M, initial_guess)[0].cpu().numpy()
    inner_boundary_np = inner_boundary[0].cpu().numpy()
    outer_boundary_np = outer_boundary[0].cpu().numpy()

fig = plt.figure()
# targets_np = targets[0].cpu().detach().numpy()
# plt.scatter(targets_np[:,0], targets_np[:,1], label="Targets",  marker='o', facecolors='none', edgecolors="g")
plt.plot(initial_guess_np[:,0], initial_guess_np[:,1], c="b", label="Initial Guess Bezier Curve")
plt.plot(outputs_np[:,0], outputs_np[:,1], c="r", label="Optimized Bezier Curve")
plt.scatter(inner_boundary_np[:,0], inner_boundary_np[:,1], label="Inner Boundary")
plt.scatter(outer_boundary_np[:,0], outer_boundary_np[:,1], label="Outer Boundary")
plt.legend()
# for i in range(pointshistory.shape[0]):
#     plt.plot(pointshistory_np[i,:,0], pointshistory_np[i,:,1])
plt.xlim(-6.0,6.0)
plt.ylim(-1.0,11.0)
plt.show()

