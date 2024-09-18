import torch.nn, torch.nn.parameter, torch.distributions
import torch.nn.functional as F
import numpy as np
import deepracing_models.math_utils as mu
import deepracing_models.math_utils.bounds_checking as bounds_checking
import deepracing_models.math_utils.dynamics as dynamics
import deepracing_models.math_utils.statistics as statistics
class BayesianFilter(torch.nn.Module):
    def __init__(self, *,
                 collision_probability_estimator : statistics.CollisionProbabilityEstimator,
                 dynamic_violation_estimator : dynamics.ExceedLimitsProbabilityEstimator,
                 bounds_checker : bounds_checking.BoundsChecker,
                 ) -> None:
        super(BayesianFilter, self).__init__()
        self.collision_probability_estimator=collision_probability_estimator
        self.dynamic_violation_estimator=dynamic_violation_estimator
        self.bounds_checker=bounds_checker
    def forward(self, *, 
                candidate_curves : torch.Tensor, candidate_curves_tstart : torch.Tensor, candidate_curves_dT : torch.Tensor,
                target_vehicle_prediction : tuple | torch.distributions.MultivariateNormal, 
                bounds_check_newton_params : dict = dict(), dynamics_check_newton_params : dict = dict()):
        Nparticles = candidate_curves.shape[0]
        candidate_curve_derivs = (candidate_curves.shape[-2]-1)*torch.diff(candidate_curves, dim=-2)/candidate_curves_dT[...,None,None]
        candidate_curve_2ndderivs = (candidate_curve_derivs.shape[-2]-1)*torch.diff(candidate_curve_derivs, dim=-2)/candidate_curves_dT[...,None,None]
        
        #Collision check
        collision_check_device = self.collision_probability_estimator.gl1d.eta.device
        collision_check_gauss_order : int = int(self.collision_probability_estimator.gl1d.eta.shape[0])
        collision_check_times : torch.Tensor = self.collision_probability_estimator.gl1d.eta.view(1,collision_check_gauss_order).expand(Nparticles, collision_check_gauss_order).to(device=candidate_curves.device)
        collision_check_positions, collision_check_buckets = mu.compositeBezierEval(candidate_curves_tstart, candidate_curves_dT, candidate_curves, collision_check_times)

        collision_check_velocities, _ = mu.compositeBezierEval(candidate_curves_tstart, candidate_curves_dT, candidate_curve_derivs, collision_check_times, idxbuckets=collision_check_buckets)
        collision_check_speeds = torch.norm(collision_check_velocities, p=2.0, dim=-1, keepdim=True)
        collision_check_tangents = collision_check_velocities/collision_check_speeds

        gauss_pts, gaussian_pdf_vals, mvn, collision_probs, overall_lambdas, overall_collision_free_probs = self.collision_probability_estimator(
           collision_check_positions.to(device=collision_check_device),  collision_check_tangents.to(device=collision_check_device), *((t.to(device=collision_check_device) for t in target_vehicle_prediction) if (type(target_vehicle_prediction)==tuple) else (target_vehicle_prediction,))
        )

        #Bounds Check
        bounds_check_gauss_order : int = int(self.bounds_checker.gl1d.eta.shape[0])
        bounds_check_times : torch.Tensor = self.bounds_checker.gl1d.eta.view(1,bounds_check_gauss_order).expand(Nparticles, bounds_check_gauss_order)
        bounds_check_positions, bounds_check_buckets = mu.compositeBezierEval(candidate_curves_tstart, candidate_curves_dT, candidate_curves, bounds_check_times)
        closest_point_r, closest_point_values, closest_point_tangents, closest_point_normals, deltas,\
        signed_distances, left_width_vals, right_width_vals, \
        specific_left_bound_violation_probs, specific_right_bound_violation_probs,\
        no_left_bound_violation_probs, no_right_bound_violation_probs = self.bounds_checker(bounds_check_positions, **bounds_check_newton_params)
        
        #Dynamics Check
        dynamics_check_gauss_order : int = int(self.dynamic_violation_estimator.gl1d.eta.shape[0])
        dynamics_check_times : torch.Tensor = self.dynamic_violation_estimator.gl1d.eta.view(1,dynamics_check_gauss_order).expand(Nparticles, dynamics_check_gauss_order)
        dynamics_check_velocities, dynamics_check_idxbuckets = mu.compositeBezierEval(candidate_curves_tstart, candidate_curves_dT, candidate_curve_derivs, dynamics_check_times)
        dynamics_check_accelerations, _ = mu.compositeBezierEval(candidate_curves_tstart, candidate_curves_dT, candidate_curve_2ndderivs, dynamics_check_times, idxbuckets=dynamics_check_idxbuckets)
        ellipse_points, ellipse_normals, origin, lat_radii, long_radii, signed_distances, specific_violation_probs, overall_within_limits_probs = \
            self.dynamic_violation_estimator(dynamics_check_velocities, dynamics_check_accelerations, **dynamics_check_newton_params)
        return (
            (
                bounds_check_positions,
                closest_point_r,
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
                mvn,
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



    