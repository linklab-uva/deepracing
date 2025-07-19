import deepracing_models.math_utils as mu
import numpy as np
from scipy.spatial.transform import Rotation
import torch
CAVSIM_TYPES = list({
    k: np.float32 for k in ["x", "y", "z", "s", "roll", "psi", "kappa", "xt", "yt", "zt", "xn", "yn", "zn", "vx", "ax"]
}.items())
def to_cavsim_cloud(control_points : torch.Tensor, delta_t : torch.Tensor, tsamp : torch.Tensor,
                    matrix_factories : dict) -> np.ndarray:
    if control_points.shape[-1] != 3:
        raise ValueError("Control points must be 3D (x, y, z)")
    kbezier = int(control_points.shape[-2]) - 1
    tstart = torch.cumsum(delta_t, 0) - delta_t#[0]
    if kbezier not in matrix_factories:
        matrix_factories[kbezier] = mu.BezierMatrixFactory(kbezier).to(tensor=control_points)
    if (kbezier-1) not in matrix_factories:
        matrix_factories[kbezier-1] = mu.BezierMatrixFactory(kbezier-1).to(tensor=control_points)
    if (kbezier-2) not in matrix_factories:
        matrix_factories[kbezier-2] = mu.BezierMatrixFactory(kbezier-2).to(tensor=control_points)


    (curve_points_samp,), idxbuckets = mu.compositeBezierEval(tstart[None], delta_t[None], control_points[None], tsamp[None], matrix_factories[kbezier])
    
    
    control_points_deriv = kbezier*torch.diff(control_points, dim=-2)/delta_t[:,None,None]
    (curve_vels_samp,), _ = mu.compositeBezierEval(tstart[None], delta_t[None], control_points_deriv[None], tsamp[None], matrix_factories[kbezier-1], idxbuckets=idxbuckets)
    
    control_points_2nddderiv = (kbezier-1)*torch.diff(control_points_deriv, dim=-2)/delta_t[:,None,None]
    (curve_accels_samp,), _ = mu.compositeBezierEval(tstart[None], delta_t[None], control_points_2nddderiv[None], tsamp[None], matrix_factories[kbezier-2], idxbuckets=idxbuckets)
    
    
    curve_speeds_samp : torch.Tensor = torch.linalg.vector_norm(curve_vels_samp, dim=-1, keepdim=False)
    curve_tangents_samp = curve_vels_samp/curve_speeds_samp[:,None]
    up = torch.as_tensor([[0.0, 0.0, 1.0],]).expand_as(curve_tangents_samp)

    curve_ax = torch.linalg.vecdot(curve_accels_samp, curve_tangents_samp)

    curve_normals_samp = torch.cross(up, curve_tangents_samp)
    curve_normals_samp = curve_normals_samp/torch.linalg.vector_norm(curve_normals_samp, ord=2, dim=-1, keepdim=True)
    # zvecs = torch.cross(curve_tangents_samp, curve_normals_samp)
    # curve_rotmats = torch.stack([curve_tangents_samp, curve_normals_samp, zvecs], dim=-1)
    # curve_rots = Rotation.from_matrix(curve_rotmats.cpu().numpy())
    # curve_euler = curve_rots.as_euler("ZYX",degrees=False)
    # curve_headings = curve_euler[:,0]
    curve_headings = torch.atan2(curve_tangents_samp[:,1], curve_tangents_samp[:,0])
    
    kappa_num = torch.linalg.vector_norm(torch.cross(curve_vels_samp, curve_accels_samp), ord=2, dim=-1)
    kappa_denom = torch.pow(curve_speeds_samp, 3)
    # kappas = kappa_num/kappa_denom
    kappas = torch.exp(torch.log(kappa_num) - torch.log(kappa_denom))
    

    svals = torch.zeros_like(kappas)
    curve_point_deltas = curve_points_samp[1:] - curve_points_samp[:-1]
    svals[1:] = torch.cumsum(torch.linalg.vector_norm(curve_point_deltas, ord=2, dim=-1), 0)

    # ["x", "y", "z", "s", "roll", "psi", "kappa", "xt", "yt", "zt", "xn", "yn", "zn", "vx", "ax"]
    rtn = np.zeros(curve_points_samp.shape[0], dtype=CAVSIM_TYPES)
    rtn["x"] = curve_points_samp[:,0].cpu().numpy()
    rtn["y"] = curve_points_samp[:,1].cpu().numpy()
    rtn["s"] = svals.cpu().numpy()
    rtn["psi"] = curve_headings.cpu().numpy()
    rtn["kappa"] = kappas.cpu().numpy()
    rtn["xt"] = curve_tangents_samp[:,0].cpu().numpy()
    rtn["yt"] = curve_tangents_samp[:,1].cpu().numpy()
    rtn["zt"] = curve_tangents_samp[:,2].cpu().numpy()
    rtn["xn"] = curve_normals_samp[:,0].cpu().numpy()
    rtn["yn"] = curve_normals_samp[:,1].cpu().numpy()
    rtn["zn"] = curve_normals_samp[:,2].cpu().numpy()
    rtn["vx"] = curve_speeds_samp.cpu().numpy()
    rtn["ax"] = curve_ax.cpu().numpy()
    return rtn
