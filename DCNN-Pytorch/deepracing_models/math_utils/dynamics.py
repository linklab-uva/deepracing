import torch, torch.nn
from deepracing_models.math_utils.interpolate import LinearInterpolator
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
