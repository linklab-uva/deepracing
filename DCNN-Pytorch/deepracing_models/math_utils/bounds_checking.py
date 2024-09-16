import torch, torch.nn
import deepracing_models.math_utils as mu
import deepracing_models.math_utils.bezier as bezier
import deepracing_models.math_utils.integrate as integrate
import deepracing_models.math_utils.interpolate as interpolate

class BoundsChecker(torch.nn.Module):
    def __init__(self, *, 
                refline_points : torch.Tensor,
                dr_samp : float,
                left_widths : torch.Tensor,
                right_widths : torch.Tensor,
                ) -> None:
        super(BoundsChecker, self).__init__()
        self.refline_helper : mu.SimplePathHelper = mu.SimplePathHelper.from_closed_path(refline_points, dr_samp)
        arclengths : torch.Tensor = self.refline_helper.__arclengths_in__.detach().clone()
        self.refline_helper.rebuild_kdtree()
        self.arclengths : torch.nn.Parameter = torch.nn.Parameter(arclengths, requires_grad=False)
        self.left_widths : torch.nn.Parameter = torch.nn.Parameter(left_widths, requires_grad=False)
        self.right_widths : torch.nn.Parameter = torch.nn.Parameter(right_widths, requires_grad=False)
        self.left_width_interp : interpolate.LinearInterpolator = interpolate.LinearInterpolator(arclengths, left_widths, requires_grad=False)
        self.right_width_interp : interpolate.LinearInterpolator = interpolate.LinearInterpolator(arclengths, right_widths, requires_grad=False)
    
    def forward(self, positions : torch.Tensor, newton_iterations: int = 0, newton_stepsize: float = 1, 
                max_step: float = 1, newton_termination_eps: float | None = 0.0001, newton_termination_delta_eps: float | None = 0.01):
        
        closest_point_r, closest_point_values, closest_point_tangents, closest_point_normals, deltas = \
            self.refline_helper.closest_point_approximate(positions, newton_iterations=newton_iterations, newton_stepsize=newton_stepsize,
                                                          max_step=max_step, newton_termination_eps=newton_termination_eps, newton_termination_delta_eps=newton_termination_delta_eps)
        normal_projections : torch.Tensor = torch.sum(deltas*closest_point_normals, dim=-1)
        left_width_vals : torch.Tensor = self.left_width_interp(closest_point_r)
        right_width_vals : torch.Tensor = self.right_width_interp(closest_point_r)
        return closest_point_r, closest_point_values, closest_point_tangents, closest_point_normals, deltas,\
                normal_projections, left_width_vals, right_width_vals
        

        