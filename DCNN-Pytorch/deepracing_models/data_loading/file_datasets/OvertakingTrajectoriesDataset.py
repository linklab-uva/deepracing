import torch
from torch.utils.data import Dataset
import numpy as np
import os
import yaml

from deepracing_models.math_utils import bezier
import deepracing_models.math_utils.collision_checking as cc
import tqdm
import time


def from_npz_dir(datadir : str, kbezier : int):
    with open(os.path.join(datadir, "metadata.yaml"),"r") as f:
        metadata = yaml.load(f, Loader=yaml.SafeLoader)
    with open(os.path.join(datadir, "data.npz"),"rb") as f:
        npdict = np.load(f)
        # data : dict[str,np.ndarray]= {k : npdict[k].copy() for k in npdict.keys()}
        rtn = OvertakingTrajectoriesDataset(npdict, metadata, kbezier)
    return rtn
class OvertakingTrajectoriesDataset(Dataset):
    def __init__(self, data_dict : dict[str, np.ndarray], metadata_dict : dict, kbezier : int):
        self.metadata_dict = metadata_dict
        self.data_dict = {k : v.copy() for k,v in data_dict.items()}
        attacker_points = torch.as_tensor(self.data_dict["attacker_pos"]).double()
        defender_points = torch.as_tensor(self.data_dict["defender_pos"]).type_as(attacker_points)
        attacker_vels = torch.as_tensor(self.data_dict["attacker_vel"]).type_as(attacker_points)
        defender_vels = torch.as_tensor(self.data_dict["defender_vel"]).type_as(attacker_points)
        tfit = torch.as_tensor(self.data_dict["delta_t"] - self.data_dict["delta_t"][0]).type_as(attacker_points)[None].expand_as(attacker_points[...,0])
        
        # print("tfit:", tfit)
        # print("tfit.shape:", tfit.shape)
        # print("defender_points.shape:", defender_points.shape)
        # print("attacker_points.shape:", attacker_points.shape)
        
        _, attacker_curves = bezier.bezierLsqfit(attacker_points, kbezier, t=tfit, P0=attacker_points[:,0], V0=attacker_vels[:,0])
        _, defender_curves = bezier.bezierLsqfit(defender_points, kbezier, t=tfit, P0=defender_points[:,0], V0=defender_vels[:,0])
        # print(attacker_curves)
        # print(defender_curves)
        # print("attacker_curves.shape:", attacker_curves.shape)
        # print("defender_curves.shape:", defender_curves.shape)
        self.data_dict["attacker_curve"] = attacker_curves.cpu().numpy()
        self.data_dict["defender_curve"] = defender_curves.cpu().numpy()
    def compute_gt(self, curve_covars : torch.Tensor, boxpoints01 : torch.Tensor):
        self.data_dict["curve_covars"] = curve_covars.cpu().numpy()
        kbezier = curve_covars.shape[0]-1
        # zeros = torch.zeros_like(torch.as_tensor(self.data_dict["attacker_curves"][0]).type_as(curve_covars))
        # offset_mvn = torch.distributions.MultivariateNormal(zeros, covariance_matrix=curve_covars)
        s_plot = torch.linspace(0.0, 1.0, steps=int(round(1.0*(2**7))), requires_grad=False).type_as(curve_covars)
        M_plot = bezier.bezierM(s_plot[None], kbezier)
        M_deriv_plot = bezier.bezierM(s_plot[None], kbezier-1)
        flip = torch.as_tensor([-1.0, 1.0]).type_as(s_plot)
        t = tqdm.tqdm(range(len(self)), desc="Computing Monte Carlo GT")
        gt_probs = torch.empty(len(self)).type_as(curve_covars)
        gt_comptimes = gt_probs.clone()
        individual_collision_probs_dense = s_plot[None].expand(gt_probs.shape[0], s_plot.shape[0]).clone()
        for idx in t:
            attacker_curve = torch.as_tensor(self.data_dict["attacker_curve"][idx]).type_as(curve_covars)[...,:-1]
            defender_curve = torch.as_tensor(self.data_dict["defender_curve"][idx]).type_as(curve_covars)[...,:-1]
            tick = time.time()
            attacker_curve_deriv = kbezier*torch.diff(attacker_curve, dim=-2)
            # print("defender_curve.shape:", defender_curve.shape)
            # print("curve_covars.shape:", curve_covars.shape)
            probabilistic_curve = torch.distributions.MultivariateNormal(defender_curve, covariance_matrix=curve_covars)
            sampled_curves = probabilistic_curve.sample([int(round(1.0*(2**11))),]).type_as(defender_curve)
            sampled_curve_derivs = kbezier*torch.diff(sampled_curves,dim=-2)

            attacker_p_plot = (M_plot@attacker_curve)[0]
            attacker_vel_plot = (M_deriv_plot@attacker_curve_deriv)[0]
            attacker_speed_plot = torch.norm(attacker_vel_plot, p=2.0, dim=-1)
            attacker_tau_plot = attacker_vel_plot/attacker_speed_plot[...,None]
            attacker_R_plot = torch.stack([attacker_tau_plot, attacker_tau_plot[...,[1,0]]], dim=-1)
            attacker_R_plot[...,0,1]*=-1.0
            attacker_boxpoints_plot = (attacker_R_plot@boxpoints01.T[None]).transpose(-2,-1) + attacker_p_plot[:,None]

            defender_Psamp_plot = M_plot@sampled_curves
            defender_vsamp_plot = M_deriv_plot@sampled_curve_derivs
            defender_speedsamp_plot = torch.norm(defender_vsamp_plot, p=2.0, dim=-1)
            defender_tausamp_plot = defender_vsamp_plot/defender_speedsamp_plot[...,None]
            defender_Rsamp_plot = torch.stack([defender_tausamp_plot, defender_tausamp_plot[...,[1,0]]*flip], dim=-1)
            boxpoints_samp_plot = (defender_Rsamp_plot@boxpoints01.T[None]).transpose(-2,-1) + defender_Psamp_plot[...,None,:]
            res_dense = cc.rectangle_intersections2(attacker_boxpoints_plot[None].expand_as(boxpoints_samp_plot), boxpoints_samp_plot)
            collision_idx_dense : torch.Tensor = res_dense.collision_idx
            individual_collision_probs_dense[idx] = torch.sum(collision_idx_dense, dim=0)/collision_idx_dense.shape[0]
            # max_probs[idx] = individual_collision_probs_dense.max()
            contain_atleast1_collision = torch.sum(collision_idx_dense, dim=1)>0
            gt_probs[idx] = contain_atleast1_collision.sum()/contain_atleast1_collision.shape[0]
            tock = time.time()
            gt_comptimes[idx] = tock - tick
        self.data_dict["individual_collision_probs_dense"] = individual_collision_probs_dense.cpu().numpy()
        self.data_dict["gt_probs"] = gt_probs.cpu().numpy()
        self.data_dict["gt_comptimes"] = gt_comptimes.cpu().numpy()
    def __len__(self):
        return self.data_dict["attacker_pos"].shape[0]

    def __getitem__(self, idx):
        
        rtn = {k : v[idx] for k,v in self.data_dict.items() if k not in {"tcurrent","delta_t","curve_covars"}}
        rtn["tcurrent"] = self.data_dict["tcurrent"][idx]
        rtn["delta_t"] = self.data_dict["delta_t"]
        rtn["curve_covars"] = self.data_dict["curve_covars"]
        rtn["track_name"] = self.metadata_dict["track_name"]

        return rtn