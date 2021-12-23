import torch
import torch.nn, torch.nn.parameter
from deepracing_models.math_utils.interpolate import LinearInterpolator
import deepracing_models.math_utils as mu
from deepracing_models.nn_models.LossFunctions import BoundaryLoss
import torch.nn.functional as F
class BayesianFilter(torch.nn.Module):
    def __init__(self, speeds: torch.Tensor, braking_limits : torch.Tensor, num_points : int, bezier_order: int, num_samples : int, \
                        inner_boundary : torch.Tensor, inner_boundary_normals : torch.Tensor, outer_boundary : torch.Tensor, outer_boundary_normals : torch.Tensor, \
                        beta_speed = 0.1, beta_ca = 1.0, beta_brake=1.0, beta_boundary=1.0, boundary_allowance=0.0, \
                        max_centripetal_acceleration=19.6):
        super(BayesianFilter,self).__init__()
        self.braking_limit_interp : LinearInterpolator = LinearInterpolator(speeds, braking_limits, requires_grad=False)
        self.num_samples : int = num_samples
        s : torch.Tensor = torch.linspace(0.0, 1.0, steps=num_points, dtype=torch.float64).unsqueeze(0)
        self.bezierM : torch.nn.parameter.Parameter = torch.nn.parameter.Parameter(mu.bezierM(s, bezier_order), requires_grad=False)
        self.bezierMderiv : torch.nn.parameter.Parameter = torch.nn.parameter.Parameter(mu.bezierM(s, bezier_order-1), requires_grad=False)
        self.bezierM2ndderiv : torch.nn.parameter.Parameter = torch.nn.parameter.Parameter(mu.bezierM(s, bezier_order-2), requires_grad=False)
        self.beta_speed : torch.nn.parameter.Parameter =  torch.nn.parameter.Parameter(torch.as_tensor(beta_speed, dtype=s.dtype), requires_grad=False)
        self.beta_boundary : torch.nn.parameter.Parameter =  torch.nn.parameter.Parameter(torch.as_tensor(beta_boundary, dtype=s.dtype), requires_grad=False)
        self.beta_brake : torch.nn.parameter.Parameter =  torch.nn.parameter.Parameter(torch.as_tensor(beta_brake, dtype=s.dtype), requires_grad=False)
        self.beta_ca : torch.nn.parameter.Parameter =  torch.nn.parameter.Parameter(torch.as_tensor(beta_ca, dtype=s.dtype), requires_grad=False)
        self.max_centripetal_acceleration = max_centripetal_acceleration

        self.boundary_allowance = boundary_allowance
        self.boundary_loss : BoundaryLoss = BoundaryLoss(time_reduction="all", batch_reduction="all", relu_type="Leaky", alpha=1.0, beta=1.0)
        self.inner_boundary : torch.nn.parameter.Parameter =  torch.nn.parameter.Parameter(torch.as_tensor(inner_boundary, dtype=s.dtype).unsqueeze(0), requires_grad=False)
        self.inner_boundary_normals : torch.nn.parameter.Parameter =  torch.nn.parameter.Parameter(torch.as_tensor(inner_boundary_normals, dtype=s.dtype).unsqueeze(0), requires_grad=False)
        self.outer_boundary : torch.nn.parameter.Parameter =  torch.nn.parameter.Parameter(torch.as_tensor(outer_boundary, dtype=s.dtype).unsqueeze(0), requires_grad=False)
        self.outer_boundary_normals : torch.nn.parameter.Parameter =  torch.nn.parameter.Parameter(torch.as_tensor(outer_boundary_normals, dtype=s.dtype).unsqueeze(0), requires_grad=False)
    
    def forward(self, curve : torch.Tensor, deltaT : float):
        curve_unsqueeze = curve.unsqueeze(0)
        curves = curve_unsqueeze+torch.randn(self.num_samples, curve.shape[0], curve.shape[1], dtype=curve.dtype, device=curve.device)
        _, v_s = mu.bezierDerivative(curves, M=self.bezierMderiv[0])
        v_t=v_s/deltaT
        speeds = torch.norm(v_t,dim=2,p=2)
        unit_tangents = v_t/speeds[:,:,None]

        average_speeds = torch.mean(speeds,dim=1)
        # max_average_speed = torch.max(average_speeds)
        # initial_speeds = speeds[:,0]
        # max_initial_speed = torch.max(initial_speeds)
        # speed_scores = torch.exp(-12.5*(1.0-(average_speeds/max_average_speed)))
        speed_scores = F.softmax(self.beta_speed*average_speeds, dim=0)
        #speed_scores = speed_scores/torch.max(speed_scores)

        _, a_s = mu.bezierDerivative(curves, M=self.bezierM2ndderiv.expand(curves.shape[0],-1,-1), order=2)
        a_t=a_s/(deltaT*deltaT)
        linear_accels = torch.sum(a_t*unit_tangents, dim=2)
        linear_accel_vecs = unit_tangents*linear_accels[:,:,None]
        braking_limits = self.braking_limit_interp(speeds)
        braking_constraint_violations = torch.clip(linear_accels - braking_limits, -1.0E9, 0.0)
        worst_braking_violations, _ = torch.min(braking_constraint_violations, dim=1)
        # mean_braking_violations = torch.mean(braking_constraint_violations, dim=1)
        braking_scores = torch.clip(torch.exp(self.beta_brake*worst_braking_violations.double()), 0.0, 1.0)
        
                
        centripetal_accel_vecs = a_t - linear_accel_vecs
        centripetal_accels = torch.norm(centripetal_accel_vecs, p=2, dim=2)
        ca_maxes, _ = torch.max(centripetal_accels, dim=1)
        max_ca_deltas = torch.relu(ca_maxes - self.max_centripetal_acceleration)
        ca_scores = torch.clip(torch.exp(-self.beta_ca*max_ca_deltas.double()), 0.0, 1.0)

        curve_points = torch.matmul(self.bezierM[0], curves)
        _, ib_distances = self.boundary_loss(curve_points, self.inner_boundary.expand(curve_points.shape[0], -1, -1), self.inner_boundary_normals.expand(curve_points.shape[0], -1, -1))
        ib_max_distances, _ = torch.max(ib_distances, dim=1)
        ib_max_distances=F.relu(ib_max_distances - self.boundary_allowance)

        _, ob_distances = self.boundary_loss(curve_points, self.outer_boundary.expand(curve_points.shape[0], -1, -1), self.outer_boundary_normals.expand(curve_points.shape[0], -1, -1))
        ob_max_distances, _ = torch.max(ob_distances, dim=1)
        ob_max_distances=F.relu(ob_max_distances - self.boundary_allowance)

        all_distances = torch.stack([ib_max_distances, ob_max_distances], dim=0)

        overall_max_distances, _ = torch.max(all_distances, dim=0)

        boundary_scores = torch.clip( torch.exp(-self.beta_boundary*overall_max_distances.double()), 1E-32, 1.0)

        score_products = ca_scores*braking_scores*boundary_scores*speed_scores
        probs = (score_products/torch.sum(score_products))

        return torch.sum(probs[:,None,None]*curves.double(), dim=0).type(self.bezierM.dtype)
