
import torch
import torch.utils
import torch.utils.data
import numpy as np
import typing
import deepracing_models
import deepracing_models.math_utils
from deepracing_models.math_utils import SimplePathHelper
from path_server.smooth_path_helper import SmoothPathHelper
from tqdm import tqdm
import torch.jit

class TrajectoryPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, timestamps : torch.Tensor,\
                 ego_vehicle_positions : torch.Tensor,\
                 ego_vehicle_velocities : torch.Tensor,\
                 target_vehicle_positions : torch.Tensor,\
                 target_vehicle_velocities : torch.Tensor,\
                 inner_boundary_helper : SimplePathHelper,\
                 outer_boundary_helper : SimplePathHelper,\
                 ego_vehicle_accelerations : torch.Tensor = None,\
                 target_vehicle_accelerations : torch.Tensor = None,
                 orient_to_inner_boundary = False):
        self.timestamps : torch.Tensor = timestamps
        self.ego_vehicle_positions : torch.Tensor = ego_vehicle_positions.clone()
        self.target_vehicle_positions : torch.Tensor = target_vehicle_positions.clone()
        self.ego_vehicle_velocities : torch.Tensor = ego_vehicle_velocities.clone()
        self.target_vehicle_velocities : torch.Tensor = target_vehicle_velocities.clone()
        self.inner_boundary_helper : SimplePathHelper = inner_boundary_helper
        self.outer_boundary_helper : SimplePathHelper = outer_boundary_helper
        if (ego_vehicle_accelerations is None) and (target_vehicle_accelerations is not None):
            raise ValueError("target_vehicle_accelerations was provided, but not ego_vehicle_accelerations. Must provide either both or neither")
        if (target_vehicle_accelerations is None) and (ego_vehicle_accelerations is not None):
            raise ValueError("ego_vehicle_accelerations was provided, but not target_vehicle_accelerations. Must provide either both or neither")
        if ego_vehicle_accelerations is not None:
            self.ego_vehicle_accelerations : torch.Tensor = ego_vehicle_accelerations.clone()
            self.target_vehicle_accelerations : torch.Tensor = target_vehicle_accelerations.clone()
        else:
            self.ego_vehicle_accelerations  = None
            self.target_vehicle_accelerations = None


        self.inner_boundary_corresponding_r : torch.Tensor = torch.zeros_like(self.target_vehicle_positions[:,0])
        self.outer_boundary_corresponding_r : torch.Tensor = self.inner_boundary_corresponding_r.clone()

        for i in tqdm(range(self.inner_boundary_corresponding_r.shape[0]), desc="Calculating closest points to the boundaries"):
            pquery : np.ndarray = self.target_vehicle_positions[i]
            _, _, r_ib, pclosest_ib  = deepracing_models.math_utils.closestPointToPath(self.inner_boundary_helper, pquery, lr = 0.5, max_iter = 5)
            _, _, r_ob, pclosest_ob  = deepracing_models.math_utils.closestPointToPath(self.inner_boundary_helper, pquery, lr = 0.5, max_iter = 5)
            self.inner_boundary_corresponding_r[i] = r_ib
            self.outer_boundary_corresponding_r[i] = r_ob
        self.sdelta = torch.linspace(0.0, 400.0, steps=30, dtype=self.target_vehicle_positions.dtype, device=self.target_vehicle_positions.device)
        self.orient_to_inner_boundary = orient_to_inner_boundary

        
    def __len__(self):
        return self.target_vehicle_positions.shape[0]-60
    def __getitem__(self, index):
        istart = index
        iend = istart + 30
        sforward_ib = self.sdelta + self.inner_boundary_corresponding_r[iend] 
        sforward_ob = self.sdelta + self.outer_boundary_corresponding_r[iend]

        ibforward, ibtangentsforward = self.inner_boundary_helper(sforward_ib)
        obforward, _ = self.outer_boundary_helper(sforward_ob)
        
        if self.orient_to_inner_boundary:
            tangent0 : torch.Tensor = ibtangentsforward[0]
        else:    
            vel0 : torch.Tensor = self.target_vehicle_velocities[iend]
            tangent0 : torch.Tensor = vel0/(torch.linalg.norm(vel0, ord=2))
        translation : torch.Tensor = self.target_vehicle_positions[iend]

        normal0 : torch.Tensor = tangent0[[1,0]]*torch.as_tensor([-1.0, 1.0], dtype=tangent0.dtype, device=tangent0.device)
        rotmat : torch.Tensor = torch.stack([tangent0, normal0], axis=1)

        rotmatinv = rotmat.transpose(0,1)
        translationinv = torch.matmul(rotmatinv, -translation)

        outdict : dict = dict()
        outdict["target_position_history"] = torch.matmul(rotmatinv, self.target_vehicle_positions[istart:iend+1].T).T + translationinv
        outdict["target_velocity_history"] = torch.matmul(rotmatinv, self.target_vehicle_velocities[istart:iend+1].T).T
        outdict["ego_position_history"] = torch.matmul(rotmatinv, self.ego_vehicle_positions[istart:iend+1].T).T + translationinv
        outdict["ego_velocity_history"] = torch.matmul(rotmatinv, self.ego_vehicle_velocities[istart:iend+1].T).T
        outdict["ibforward_local"] = torch.matmul(rotmatinv, ibforward.T).T + translationinv
        outdict["obforward_local"] = torch.matmul(rotmatinv, obforward.T).T + translationinv
        if self.target_vehicle_accelerations is not None:
            outdict["ego_acceleration_history"] = torch.matmul(rotmatinv, self.ego_vehicle_accelerations[istart:iend+1].T).T
            outdict["target_acceleration_history"] = torch.matmul(rotmatinv, self.target_vehicle_accelerations[istart:iend+1].T).T
        return outdict
        
        