import torch, torch.nn, torch.nn.functional as F
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
                gauss_order : int,
                dT : float,
                stdev : float = 1.25,
                squared_distances : bool = True, 
                levels : int = None
                ) -> None:
        super(BoundsChecker, self).__init__()
        self.refline_helper : mu.SimplePathHelper = mu.SimplePathHelper.from_closed_path(refline_points, dr_samp)
        arclengths : torch.Tensor = self.refline_helper.__arclengths_in__.detach().clone()
        self.rebuild_kdtree(squared_distances=squared_distances, levels=levels)
        self.stdev_factor : torch.nn.Parameter = torch.nn.Parameter(1.0/(torch.as_tensor(2.0).sqrt()*stdev), requires_grad=False)
        self.arclengths : torch.nn.Parameter = torch.nn.Parameter(arclengths, requires_grad=False)
        self.left_widths : torch.nn.Parameter = torch.nn.Parameter(left_widths, requires_grad=False)
        self.right_widths : torch.nn.Parameter = torch.nn.Parameter(right_widths, requires_grad=False)
        self.left_width_interp : interpolate.LinearInterpolator = interpolate.LinearInterpolator(arclengths, left_widths, requires_grad=False)
        self.right_width_interp : interpolate.LinearInterpolator = interpolate.LinearInterpolator(arclengths, right_widths, requires_grad=False)
        self.gl1d : integrate.GaussLegendre1D = integrate.GaussLegendre1D(gauss_order, interval=[0, dT], requires_grad=False)
    def rebuild_kdtree(self, squared_distances : bool = True, levels : int = None):
        self.refline_helper.rebuild_kdtree(squared_distances=squared_distances, levels=levels)
        
    def forward(self, positions : torch.Tensor, newton_iterations: int = 0, newton_stepsize: float = 1, 
                max_step: float = 1, newton_termination_eps: float | None = 0.0001, newton_termination_delta_eps: float | None = 0.01):
        
        closest_point_r, closest_point_values, closest_point_tangents, closest_point_normals, deltas = \
            self.refline_helper.closest_point_approximate(positions, newton_iterations=newton_iterations, newton_stepsize=newton_stepsize,
                                                          max_step=max_step, newton_termination_eps=newton_termination_eps, newton_termination_delta_eps=newton_termination_delta_eps)
        signed_distances : torch.Tensor = torch.sum(deltas*closest_point_normals, dim=-1)
        left_width_vals : torch.Tensor = self.left_width_interp(closest_point_r)
        right_width_vals : torch.Tensor = self.right_width_interp(closest_point_r)

        specific_left_bound_violation_probs = torch.special.erf(F.relu(signed_distances - left_width_vals)*self.stdev_factor)
        specific_right_bound_violation_probs = torch.special.erf(F.relu(right_width_vals - signed_distances)*self.stdev_factor)
        left_bound_overall_lambdas : torch.Tensor = self.gl1d(specific_left_bound_violation_probs)
        right_bound_overall_lambdas : torch.Tensor = self.gl1d(specific_right_bound_violation_probs)

        no_left_bound_violation_probs = torch.exp(-left_bound_overall_lambdas)
        no_right_bound_violation_probs = torch.exp(-right_bound_overall_lambdas)

        return closest_point_r, closest_point_values, closest_point_tangents, closest_point_normals, deltas,\
                signed_distances, left_width_vals, right_width_vals, specific_left_bound_violation_probs, \
                    specific_right_bound_violation_probs, no_left_bound_violation_probs, no_right_bound_violation_probs
        

        