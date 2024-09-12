import numpy as np
import torch, torch.nn
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
                 requires_grad=False) -> None:
        super(ExceedLimitsProbabilityEstimator, self).__init__()
        self.dynamics_interp : DynamicsInterp = DynamicsInterp(
                braking_speeds, braking_maxvals,
                longaccel_speeds, longaccel_maxvals,
                lataccel_speeds, lataccel_maxvals,
                requires_grad=requires_grad)
        self.tangent_to_normal_rotmat : torch.nn.Parameter = torch.nn.Parameter(torch.as_tensor([
            [0.0, -1.0],
            [1.0,  0.0]
        ]), requires_grad=requires_grad)
    def forward(self, velocities : torch.Tensor, accels : torch.Tensor, 
                newton_iterations = 20, newton_stepsize = 0.5, max_step=1.75*_pi_180, 
                newton_termination_eps = 1E-5, newton_termination_delta_eps = 1E-6):
        speeds : torch.Tensor = torch.norm(velocities, p=2.0, dim=-1, keepdim=True)
        tangents = velocities/speeds
        normals = (self.tangent_to_normal_rotmat @ tangents[...,None])[...,0]
        origin, lat_radii, long_radii = self.dynamics_interp(speeds.squeeze(-1))
        # return None, origin, lat_radii, long_radii
        long_accels = torch.sum(accels*tangents, dim=-1, keepdim=True)
        lat_accels = torch.sum(accels*normals, dim=-1, keepdim=True)
        # print("long_accels", long_accels)
        # print("lat_accels", lat_accels)
        both_accels = torch.cat([lat_accels, long_accels], dim=-1)
        both_accels_offset = both_accels - origin
        thetas = torch.atan2(both_accels_offset[...,1], both_accels_offset[...,0])
        ellipse_points : torch.Tensor = torch.stack([lat_radii*torch.cos(thetas), long_radii*torch.sin(thetas)], dim=-1) + origin
        radii_ratio = (long_radii/lat_radii)
        gamma = -radii_ratio/torch.tan(thetas)#*torch.cos(thetas)/torch.sin(thetas)
        alpha  = torch.arctan(gamma)
        tau = torch.stack([torch.cos(alpha), torch.sin(alpha)], dim=-1)
        

        for _ in range(newton_iterations):
            deltas = both_accels - ellipse_points
            # squared_distances : torch.Tensor = torch.sum(torch.square(deltas), dim=-1)
            # squared_distance_derivs : torch.Tensor = torch.sum(deltas*torch.stack([lat_radii*torch.sin(thetas), -long_radii*torch.cos(thetas)], dim=-1), dim=-1)
            # thetas-=torch.clip(newton_stepsize*(squared_distances/squared_distance_derivs).nan_to_num(nan=0.0, posinf=max_step, neginf=-max_step), -max_step, max_step)
            radii_ratio = (long_radii/lat_radii)
            dotprods = torch.sum(deltas*tau, dim=-1)
            dgamma_dtheta = radii_ratio/torch.square(torch.sin(thetas))
            dalpha_dtheta = dgamma_dtheta/(torch.square(gamma) + 1)
            dtau_dtheta = torch.stack([-torch.sin(alpha)*dalpha_dtheta, torch.cos(alpha)*dalpha_dtheta], dim=-1)
            ddelta_dtheta = torch.stack([lat_radii*torch.sin(thetas), -long_radii*torch.cos(thetas)], dim=-1)
            dotprod_deriv = 0.5*(deltas[...,0]*dtau_dtheta[...,0] + ddelta_dtheta[...,0]*tau[...,0] +\
                            deltas[...,1]*dtau_dtheta[...,1] + ddelta_dtheta[...,1]*tau[...,1])
            theta_deltas = torch.clip(newton_stepsize*(dotprods/dotprod_deriv), -max_step, max_step)
            thetas-=theta_deltas
            gamma = -radii_ratio/torch.tan(thetas)
            alpha  = torch.arctan(gamma)
            tau = torch.stack([torch.cos(alpha), torch.sin(alpha)], dim=-1)
            ellipse_points = torch.stack([lat_radii*torch.cos(thetas), long_radii*torch.sin(thetas)], dim=-1) + origin
            if torch.all(torch.abs(dotprods)<newton_termination_eps):
                break
            if torch.all(torch.abs(theta_deltas)<newton_termination_delta_eps):
                break
        ellipse_normals = (self.tangent_to_normal_rotmat @ tau[...,None])[...,0]
        origin_deltas = ellipse_points - origin
        ellipse_normals*=torch.sign(torch.sum(ellipse_normals*origin_deltas, dim=-1))[...,None]
        return ellipse_points, ellipse_normals, origin, lat_radii, long_radii

