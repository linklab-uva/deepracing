import torch
import torch.nn, torch.nn.parameter
from deepracing_models.math_utils.interpolate import LinearInterpolator
import deepracing_models.math_utils as mu
class BayesianFilter(torch.nn.Module):
    def __init__(self, speeds: torch.Tensor, braking_limits : torch.Tensor, num_points : int, bezier_order: int, num_samples : int, \
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
        self.beta_ca : torch.nn.parameter.Parameter =  torch.nn.parameter.Parameter(torch.as_tensor(beta_ca, dtype=torch.float64), requires_grad=False)
        self.max_centripetal_acceleration = max_centripetal_acceleration
    def forward(self, curve : torch.Tensor, deltaT : float):
        curve_unsqueeze = curve.unsqueeze(0)
        curves = curve_unsqueeze+torch.randn_like(curve_unsqueeze.expand(self.num_samples,-1,-1))
        _, v_s = mu.bezierDerivative(curves, M=self.bezierMderiv[0])
        v_t=v_s/deltaT
        speeds = torch.norm(v_t,dim=2,p=2)
        unit_tangents = v_t/speeds[:,:,None]

        _, a_s = mu.bezierDerivative(curves, M=self.bezierM2ndderiv.expand(curves.shape[0],-1,-1), order=2)
        a_t=a_s/(deltaT*deltaT)
        linear_accels = torch.sum(a_t*unit_tangents, dim=2)
        linear_accel_vecs = unit_tangents*linear_accels[:,:,None]
        braking_limits = self.braking_limit_interp(speeds)
        braking_constraint_violations = torch.clip(linear_accels - braking_limits, -1.0E9, 0.0)
        # worst_braking_violations, _ = torch.min(braking_constraint_violations, dim=1)
        # braking_scores = torch.clip(torch.exp(2.0*worst_braking_violations.double()), 0.0, 1.0)
        mean_braking_violations = torch.mean(braking_constraint_violations, dim=1)
        # braking_scores = torch.clip(torch.exp(2.0*mean_braking_violations.double()), 0.0, 1.0)
        braking_scores = torch.clip(torch.exp(self.beta_brake*mean_braking_violations.double()), 0.0, 1.0)
        
                
        centripetal_accel_vecs = a_t - linear_accel_vecs
        centripetal_accels = torch.norm(centripetal_accel_vecs, p=2, dim=2)
        # centripetal_accels[centripetal_accels!=centripetal_accels] = 0.0
        ca_maxes, _ = torch.max(centripetal_accels, dim=1)
        max_ca_deltas = torch.relu(ca_maxes - self.max_centripetal_acceleration)
        # max_ca_deltas, _ = torch.max(ca_deltas, dim=1)
        # ca_scores = torch.clip(torch.exp(-2.0*max_ca_deltas.double()), 0.0, 1.0)
        ca_scores = torch.clip(torch.exp(self.beta_ca*max_ca_deltas.double()), 0.0, 1.0)

        score_products = ca_scores*braking_scores#*boundary_scores*speed_scores
        probs = (score_products/torch.sum(score_products))

        return torch.sum(probs[:,None,None]*curves.double(), dim=0).type(self.bezierM.dtype)
