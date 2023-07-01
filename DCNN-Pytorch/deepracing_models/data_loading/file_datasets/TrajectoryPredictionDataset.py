
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
                 target_vehicle_accelerations : torch.Tensor = None):
        self.timestamps : torch.Tensor = timestamps
        self.ego_vehicle_positions : torch.Tensor = ego_vehicle_positions
        self.target_vehicle_positions : torch.Tensor = target_vehicle_positions
        self.ego_vehicle_velocities : torch.Tensor = ego_vehicle_velocities 
        self.target_vehicle_velocities : torch.Tensor = target_vehicle_velocities
        self.inner_boundary_helper : SimplePathHelper = inner_boundary_helper
        self.outer_boundary_helper : SimplePathHelper = outer_boundary_helper
        if (ego_vehicle_accelerations is None) and (target_vehicle_accelerations is not None):
            raise ValueError("target_vehicle_accelerations was provided, but not ego_vehicle_accelerations. Must provide either both or neither")
        if (target_vehicle_accelerations is None) and (ego_vehicle_accelerations is not None):
            raise ValueError("ego_vehicle_accelerations was provided, but not target_vehicle_accelerations. Must provide either both or neither")
        self.ego_vehicle_accelerations : torch.Tensor = ego_vehicle_accelerations
        self.target_vehicle_accelerations : torch.Tensor = target_vehicle_accelerations

        self.inner_boundary_corresponding_r : torch.Tensor = torch.zeros_like(self.target_vehicle_positions[:,0])
        self.outer_boundary_corresponding_r : torch.Tensor = self.inner_boundary_corresponding_r.clone()

        for i in tqdm(range(self.inner_boundary_corresponding_r.shape[0]), desc="Calculating closest points to the boundaries"):
            pquery : np.ndarray = self.target_vehicle_positions[i]
            _, _, r_ib, pclosest_ib  = deepracing_models.math_utils.closestPointToPath(self.inner_boundary_helper, pquery, lr = 0.5, max_iter = 5)
            _, _, r_ob, pclosest_ob  = deepracing_models.math_utils.closestPointToPath(self.inner_boundary_helper, pquery, lr = 0.5, max_iter = 5)
            self.inner_boundary_corresponding_r[i] = r_ib
            self.outer_boundary_corresponding_r[i] = r_ob

        
    def __len__(self):
        return self.target_vehicle_positions.shape[0]-60
    def __getitem__(self, index):
        istart = index
        iend = istart + 30
        sdelta = torch.linspace(0.0, 400.0, steps=30, dtype=self.target_vehicle_positions.dtype, device=self.target_vehicle_positions.device)
        sforward_ib = sdelta + self.inner_boundary_corresponding_r[iend] 
        sforward_ob = sdelta + self.outer_boundary_corresponding_r[iend]

        ibforward, ibtangentsforward = self.inner_boundary_helper(sforward_ib)
        obforward, _ = self.outer_boundary_helper(sforward_ob)
        
        tangent0 : torch.Tensor = ibtangentsforward[0]
        normal0 : torch.Tensor = tangent0[[1,0]]
        normal0[0]*=-1.0

        rotmat : torch.Tensor = torch.stack([tangent0, normal0], axis=1)
        translation : torch.Tensor = ibforward[0]

        rotmatinv = rotmat.t()
        translationinv = torch.matmul(rotmatinv, -translation)

        position_history = torch.matmul(rotmatinv, self.ego_vehicle_positions[istart:iend].T).T + translationinv
        velocity_history = torch.matmul(rotmatinv, self.ego_vehicle_velocities[istart:iend].T).T

        ibforward_local = torch.matmul(rotmatinv, ibforward.T).T + translationinv
        obforward_local = torch.matmul(rotmatinv, obforward.T).T + translationinv



        return position_history, velocity_history, ibforward_local, obforward_local 
        
        