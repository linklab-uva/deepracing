import torch.nn, torch.nn.parameter, torch.distributions
import torch.nn.functional as F
import numpy as np
import deepracing_models.math_utils as mu
import deepracing_models.math_utils.bezier as bezier
import deepracing_models.math_utils.bounds_checking as bounds_checking
import deepracing_models.math_utils.dynamics as dynamics
import deepracing_models.math_utils.statistics as statistics
class BayesianFilter(torch.nn.Module):
    def __init__(self, *,
                 collision_probability_estimator : statistics.CollisionProbabilityEstimator,
                 dynamic_violation_estimator : dynamics.ExceedLimitsProbabilityEstimator,
                 bounds_checker : bounds_checking.BoundsChecker,
                 bezier_order : int = 3,
                 ) -> None:
        super(BayesianFilter, self).__init__()
        self.collision_probability_estimator=collision_probability_estimator
        self.dynamic_violation_estimator=dynamic_violation_estimator
        self.bounds_checker=bounds_checker
        self.matrix_factory = bezier.BezierMatrixFactory(bezier_order)
        self.derivative_matrix_factory = bezier.BezierMatrixFactory(bezier_order - 1)
        self.second_derivative_matrix_factory = bezier.BezierMatrixFactory(bezier_order - 2)
        self.minusonehalf = torch.nn.Parameter(torch.as_tensor(-0.5), requires_grad=False)
        self.minusone = torch.nn.Parameter(torch.as_tensor(-1.0), requires_grad=False)
        flip = torch.ones(2)
        flip[0] = -1.0
        self.flip = torch.nn.Parameter(flip, requires_grad=False)
    # @torch.jit.script
    def forward(self, 
                candidate_curves : torch.Tensor, candidate_curves_tstart : torch.Tensor, candidate_curves_dT : torch.Tensor,
                target_means : torch.Tensor, target_stdev_inverse_matrices : torch.Tensor, target_logstdevs : torch.Tensor,
                bounds_check_newton_params : dict | None, dynamics_check_newton_params : dict | None):
        Nparticles = candidate_curves.shape[0]
        candidate_curve_derivs = (candidate_curves.shape[-2]-1)*torch.diff(candidate_curves, dim=-2)/candidate_curves_dT[...,None,None]
        candidate_curve_2ndderivs = (candidate_curve_derivs.shape[-2]-1)*torch.diff(candidate_curve_derivs, dim=-2)/candidate_curves_dT[...,None,None]
        
        #Collision check
        # collision_check_device = self.collision_probability_estimator.gl1d.eta.device
        collision_check_gauss_order : int = int(self.collision_probability_estimator.gl1d.eta.shape[0])
        collision_check_times : torch.Tensor = self.collision_probability_estimator.gl1d.eta.view(1,collision_check_gauss_order).expand(Nparticles, collision_check_gauss_order).to(device=candidate_curves.device)
        collision_check_positions, collision_check_buckets = mu.compositeBezierEval(candidate_curves_tstart, candidate_curves_dT, candidate_curves, collision_check_times, self.matrix_factory)

        collision_check_velocities, _ = mu.compositeBezierEval(candidate_curves_tstart, candidate_curves_dT, candidate_curve_derivs, collision_check_times, self.derivative_matrix_factory, idxbuckets=collision_check_buckets)
        collision_check_speeds = torch.norm(collision_check_velocities, p=2.0, dim=-1, keepdim=True)
        collision_check_speed_inverses = torch.pow(collision_check_speeds, self.minusone)
        # collision_check_tangents : torch.Tensor = collision_check_velocities/collision_check_speeds
        collision_check_tangents = collision_check_velocities*collision_check_speed_inverses#torch.pow(collision_check_speeds, self.minusone)
        collision_check_normals = collision_check_tangents[...,[1,0]] * self.flip[None,None]
        # collision_check_normals : torch.Tensor = collision_check_tangents[...,[1,0]].clone()
        # collision_check_normals[...,0]*=-1.0
        collision_check_rotmats = torch.stack([collision_check_tangents, collision_check_normals], dim=-1)

        gauss_pts, gaussian_pdf_vals, _, collision_probs, overall_lambdas, overall_collision_free_probs = self.collision_probability_estimator(
            target_means, target_stdev_inverse_matrices, target_logstdevs,
           collision_check_rotmats,  collision_check_positions 
        )

        #Bounds Check
        bounds_check_gauss_order : int = int(self.bounds_checker.gl1d.eta.shape[0])
        bounds_check_times : torch.Tensor = self.bounds_checker.gl1d.eta.view(1,bounds_check_gauss_order).expand(Nparticles, bounds_check_gauss_order)
        bounds_check_positions, bounds_check_buckets = mu.compositeBezierEval(candidate_curves_tstart, candidate_curves_dT, candidate_curves, bounds_check_times, self.matrix_factory)
        closest_point_r, closest_point_values, closest_point_tangents, closest_point_normals, deltas,\
        signed_distances, left_width_vals, right_width_vals, \
        specific_left_bound_violation_probs, specific_right_bound_violation_probs,\
        no_left_bound_violation_probs, no_right_bound_violation_probs = self.bounds_checker(bounds_check_positions, 
                                                                                            **(bounds_check_newton_params if bounds_check_newton_params is not None else dict()))
        
        #Dynamics Check
        dynamics_check_gauss_order : int = int(self.dynamic_violation_estimator.gl1d.eta.shape[0])
        dynamics_check_times : torch.Tensor = self.dynamic_violation_estimator.gl1d.eta.view(1,dynamics_check_gauss_order).expand(Nparticles, dynamics_check_gauss_order)
        dynamics_check_velocities, dynamics_check_idxbuckets = mu.compositeBezierEval(candidate_curves_tstart, candidate_curves_dT, candidate_curve_derivs, dynamics_check_times, self.derivative_matrix_factory)
        dynamics_check_accelerations, _ = mu.compositeBezierEval(candidate_curves_tstart, candidate_curves_dT, candidate_curve_2ndderivs, dynamics_check_times, self.second_derivative_matrix_factory, idxbuckets=dynamics_check_idxbuckets)
        ellipse_points, ellipse_normals, origin, lat_radii, long_radii, signed_distances, specific_violation_probs, overall_within_limits_probs = \
            self.dynamic_violation_estimator(dynamics_check_velocities, dynamics_check_accelerations, **(dynamics_check_newton_params if dynamics_check_newton_params is not None else dict()))
        return (
            (
                closest_point_r,
                bounds_check_positions,
                closest_point_values, 
                closest_point_tangents,
                closest_point_normals,
                deltas,
                signed_distances,
                left_width_vals,
                right_width_vals,
                specific_left_bound_violation_probs,
                specific_right_bound_violation_probs,
                no_left_bound_violation_probs,
                no_right_bound_violation_probs,
           ),
            (
                gauss_pts,
                gaussian_pdf_vals,
                collision_check_positions,
                collision_probs,
                overall_lambdas,
                overall_collision_free_probs
            ),
            (
                ellipse_points,
                ellipse_normals,
                origin,
                lat_radii,
                long_radii,
                signed_distances,
                specific_violation_probs,
                overall_within_limits_probs
            ),
        )
        # return {
        #     "bounds_check":
        #     {
        #         "checked_positions" : bounds_check_positions,
        #         "closest_point_r" : closest_point_r,
        #         "closest_point_values" : closest_point_values,
        #         "closest_point_tangents" : closest_point_tangents,
        #         "closest_point_normals" : closest_point_normals,
        #         "deltas" : deltas,
        #         "signed_distances" : signed_distances,
        #         "left_width_vals" : left_width_vals,
        #         "right_width_vals" : right_width_vals,
        #         "specific_left_bound_probabilities" : specific_left_bound_violation_probs,
        #         "specific_right_bound_probabilities" : specific_right_bound_violation_probs,
        #         "no_left_bound_violation_probs" : no_left_bound_violation_probs,
        #         "no_right_bound_violation_probs" : no_right_bound_violation_probs,
        #     },
        #     "collision_check":
        #     {
        #         "mvn" : mvn,
        #         "specific_probabilities" : collision_probs,
        #         "overall_lambdas" : overall_lambdas,
        #         "overall_probabilities" : overall_collision_free_probs
        #     },
        #     "dynamics_check":
        #     {
        #         "ellipse_points" : ellipse_points,
        #         "ellipse_normals" : ellipse_normals,
        #         "ellipse_origins" : origin,
        #         "lat_radii" : lat_radii,
        #         "long_radii" : long_radii,
        #         "signed_distances" : signed_distances,
        #         "specific_probabilities" : specific_violation_probs,
        #         "overall_probabilities" : overall_violation_probs
        #     },
        # }



    