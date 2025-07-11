import numpy as np
import torch, torch.nn, torch.nn.functional as F
from deepracing_models.math_utils.integrate import GaussLegendre1D
from deepracing_models.math_utils.interpolate import LinearInterpolator
_2pi = 2.0*np.pi
_pi_180 = np.pi/180.0
_180_pi = 180.0/np.pi
class DynamicsInterp(torch.nn.Module):
    def __init__(self, 
                 braking_speeds : torch.Tensor, braking_maxvals : torch.Tensor,
                 longaccel_speeds : torch.Tensor, longaccel_maxvals : torch.Tensor,
                 lataccel_speeds : torch.Tensor, lataccel_maxvals : torch.Tensor,
                 requires_grad=False) -> None:
        super(DynamicsInterp, self).__init__()
        idx_sort = torch.argsort(braking_speeds)
        self.braking_interp : LinearInterpolator = LinearInterpolator(braking_speeds[idx_sort], braking_maxvals[idx_sort], requires_grad=requires_grad)

        idx_sort = torch.argsort(longaccel_speeds)
        self.longaccel_interp : LinearInterpolator = LinearInterpolator(longaccel_speeds[idx_sort], longaccel_maxvals[idx_sort], requires_grad=requires_grad)

        idx_sort = torch.argsort(lataccel_speeds)
        self.lataccel_interp : LinearInterpolator = LinearInterpolator(lataccel_speeds[idx_sort], lataccel_maxvals[idx_sort], requires_grad=requires_grad)
    

    def forward(self, speeds_eval : torch.Tensor):
        max_braking : torch.Tensor = self.braking_interp(speeds_eval)
        max_longaccel : torch.Tensor = self.longaccel_interp(speeds_eval)
        max_lataccel : torch.Tensor = self.lataccel_interp(speeds_eval)
        long_midpoint = 0.5*(max_braking + max_longaccel)
        origin = torch.stack([torch.zeros_like(max_lataccel),long_midpoint], dim=-1)
        long_radius = max_longaccel - long_midpoint
        lat_radius = max_lataccel
        return origin, lat_radius, long_radius
class ExceedLimitsProbabilityEstimator(torch.nn.Module):
    def __init__(self, 
                 braking_speeds : torch.Tensor, braking_maxvals : torch.Tensor,
                 longaccel_speeds : torch.Tensor, longaccel_maxvals : torch.Tensor,
                 lataccel_speeds : torch.Tensor, lataccel_maxvals : torch.Tensor,
                 gauss_order : int,
                 dT : float,
                 stdev : float = 1.75,
                 alpha : float = 0.1,
                 newton_iterations = 20,
                 newton_stepsize = 1.0,
                 max_step=1.75*_pi_180, 
                 requires_grad=False) -> None:
        super(ExceedLimitsProbabilityEstimator, self).__init__()
        self.dynamics_interp : DynamicsInterp = DynamicsInterp(
                braking_speeds, braking_maxvals,
                longaccel_speeds, longaccel_maxvals,
                lataccel_speeds, lataccel_maxvals,
                requires_grad=requires_grad)

        self.gl1d = GaussLegendre1D(gauss_order, interval=[0, dT], requires_grad=False)
        self.stdev_factor : torch.nn.Parameter = torch.nn.Parameter(1.0/(torch.as_tensor(2.0).sqrt()*stdev), requires_grad=False)
        self.alpha = torch.nn.Parameter(torch.as_tensor(alpha), requires_grad=False)
        Rflip = torch.zeros([2,2]).float()
        Rflip[0,1]=-1.0
        Rflip[1,0]=1.0
        self.Rflip_tangent = torch.nn.Parameter(Rflip, requires_grad=False)
        # Rflip[0,1]=1.0
        # Rflip[1,0]=-1.0
        self.Rflip_ellipse = torch.nn.Parameter(-Rflip, requires_grad=False)
        self.newton_stepsizes = torch.nn.Parameter(torch.ones(newton_iterations, dtype=torch.float32), requires_grad=False)
        self.max_step = torch.nn.Parameter(torch.as_tensor(max_step, dtype=torch.float32), requires_grad=False)

    def forward(self, velocities : torch.Tensor, accels : torch.Tensor):
                # newton_termination_eps : float | None = 1E-4, newton_termination_delta_eps : float | None = .1*np.pi/180.0):
        speeds : torch.Tensor = torch.linalg.vector_norm(velocities, dim=-1, keepdim=True)
        # velocity_signs = torch.sign(velocities)
        # log_tangents = torch.log(velocities) - torch.log(speeds)
        tangents = velocities/speeds
        # normals = tangents[...,[1,0]].clone()
        # normals[...,0]*=-1.0
        normals = (self.Rflip_tangent @ tangents.unsqueeze(-1)).squeeze(-1)
        origin, lat_radii, long_radii = self.dynamics_interp(speeds.squeeze(-1))
        # return None, origin, lat_radii, long_radii
        # long_accels = torch.sum(accels*tangents, dim=-1, keepdim=True)
        # lat_accels = torch.sum(accels*normals, dim=-1, keepdim=True)
        # print("long_accels", long_accels)
        # print("lat_accels", lat_accels)
        both_accels = torch.empty_like(tangents)
        both_accels[...,0] = torch.sum(accels*normals, dim=-1, keepdim=False)
        both_accels[...,1] = torch.sum(accels*tangents, dim=-1, keepdim=False)
        # torch.cat([lat_accels, long_accels], dim=-1)
        both_accels_centered = both_accels - origin
        thetas = torch.atan2(both_accels_centered[...,1], both_accels_centered[...,0])
        costheta = torch.cos(thetas)
        sintheta = torch.sin(thetas)
        # ellipse_points : torch.Tensor = torch.stack([lat_radii*torch.cos(thetas), long_radii*torch.sin(thetas)], dim=-1) + origin
        # radii_ratio = (torch.log(long_radii) - torch.log(lat_radii)).exp()
        ellipse_points_centered : torch.Tensor = torch.stack([lat_radii*costheta, long_radii*sintheta], dim=-1)#.clone()
        centered_deltas = both_accels_centered - ellipse_points_centered
        lat_radii_squared = torch.square(lat_radii)
        long_radii_squared = torch.square(long_radii)

        for idx in range(self.newton_stepsizes.shape[0]):
            newton_stepsize = self.newton_stepsizes[idx]
            deltax = centered_deltas[...,0]
            deltay = centered_deltas[...,1]
            dfunc_dtheta =    (deltax*lat_radii*sintheta)-(deltay*long_radii*costheta)
            
            d2func_dtheta2 =  (deltax*lat_radii*costheta) + (lat_radii_squared*torch.square(sintheta)) +\
                              (deltay*long_radii*sintheta) + (long_radii_squared*torch.square(costheta)) 
            theta_deltas = torch.clip(newton_stepsize*(dfunc_dtheta/d2func_dtheta2), -self.max_step, self.max_step)
            thetas -= theta_deltas
            torch.cos(thetas, out=costheta)
            torch.sin(thetas, out=sintheta)
            # ellipse_points_centered[...,0]=lat_radii*costheta
            # ellipse_points_centered[...,1]=long_radii*sintheta
            torch.stack([lat_radii*costheta, long_radii*sintheta], dim=-1, out=ellipse_points_centered)
            # torch.multiply(lat_radii, costheta, out=ellipse_points_centered[...,0])
            # torch.multiply(long_radii, sintheta, out=ellipse_points_centered[...,1])
            torch.sub(both_accels_centered, ellipse_points_centered, out=centered_deltas)
            

        tau = torch.stack([-lat_radii*sintheta, long_radii*costheta], dim=-1)
        tau /= torch.norm(tau, p=2.0, dim=-1, keepdim=True)
        ellipse_normals = (self.Rflip_ellipse@tau.unsqueeze(-1)).squeeze(-1)
        # ellipse_normals = tau[...,[1,0]].clone()
        # ellipse_normals[...,1]*=-1.0
        ellipse_normals*=torch.sign(torch.sum(ellipse_normals*ellipse_points_centered, dim=-1))[...,None]
        signed_distances = torch.sum(centered_deltas*ellipse_normals, dim=-1)
        specific_violation_probs = torch.special.erf(F.relu(signed_distances)*self.stdev_factor)
        specific_noviolation_probs = 1.0 - specific_violation_probs       
        # odds_ratios = torch.exp(torch.log(specific_violation_probs) - torch.log(specific_noviolation_probs)) 
        odds_ratios = specific_violation_probs/specific_noviolation_probs
        overall_lambdas = self.gl1d(self.alpha*specific_violation_probs + (1-self.alpha)*odds_ratios)
        overall_within_limits_probs = torch.exp(-overall_lambdas)
        ellipse_points = ellipse_points_centered + origin
        return ellipse_points, ellipse_normals, origin, lat_radii, long_radii, signed_distances, specific_violation_probs, overall_within_limits_probs

