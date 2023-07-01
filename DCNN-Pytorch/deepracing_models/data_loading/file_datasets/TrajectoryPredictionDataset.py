
import torch
import torch.utils
import torch.utils.data
import numpy as np
import typing
import deepracing_models
from deepracing_models.math_utils import CompositeBezierCurve
from path_server.smooth_path_helper import SmoothPathHelper
from tqdm import tqdm

class TrajectoryPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, timestamps : torch.Tensor,\
                 ego_vehicle_positions : torch.Tensor,\
                 ego_vehicle_velocities : torch.Tensor,\
                 target_vehicle_positions : torch.Tensor,\
                 target_vehicle_velocities : torch.Tensor,\
                 inner_boundary_helper : SmoothPathHelper,\
                 outer_boundary_helper : SmoothPathHelper,\
                 centerline_helper : SmoothPathHelper,\
                 ego_vehicle_accelerations : torch.Tensor = None,\
                 target_vehicle_accelerations : torch.Tensor = None):
        self.timestamps : torch.Tensor = timestamps
        self.ego_vehicle_positions : torch.Tensor = ego_vehicle_positions
        self.target_vehicle_positions : torch.Tensor = target_vehicle_positions
        self.ego_vehicle_velocities : torch.Tensor = ego_vehicle_velocities 
        self.target_vehicle_velocities : torch.Tensor = target_vehicle_velocities
        self.inner_boundary_helper : SmoothPathHelper = inner_boundary_helper
        self.outer_boundary_helper : SmoothPathHelper = outer_boundary_helper
        self.centerline_helper : SmoothPathHelper = centerline_helper
        if (ego_vehicle_accelerations is None) and (target_vehicle_accelerations is not None):
            raise ValueError("target_vehicle_accelerations was provided, but not ego_vehicle_accelerations. Must provide either both or neither")
        if (target_vehicle_accelerations is None) and (ego_vehicle_accelerations is not None):
            raise ValueError("ego_vehicle_accelerations was provided, but not target_vehicle_accelerations. Must provide either both or neither")
        self.ego_vehicle_accelerations : torch.Tensor = ego_vehicle_accelerations
        self.target_vehicle_accelerations : torch.Tensor = target_vehicle_accelerations

        self.inner_boundary_corresponding_r : torch.Tensor = torch.zeros_like(self.ego_vehicle_positions[:,0])
        self.outer_boundary_corresponding_r : torch.Tensor = self.inner_boundary_corresponding_r.clone()
        self.centerline_corresponding_r : torch.Tensor = self.inner_boundary_corresponding_r.clone()

        for i in tqdm(range(self.inner_boundary_corresponding_r.shape[0]), desc="Calculating closest points to the boundaries"):
            pquery_numpy : np.ndarray = self.ego_vehicle_positions[i].cpu().numpy()
            r_closest_inner_boundary, _ = self.inner_boundary_helper.closest_point(pquery_numpy)
            r_closest_outer_boundary, _ = self.outer_boundary_helper.closest_point(pquery_numpy)
            r_closest_centerline, _ = self.centerline_helper.closest_point(pquery_numpy)
            self.inner_boundary_corresponding_r[i] = r_closest_inner_boundary
            self.outer_boundary_corresponding_r[i] = r_closest_outer_boundary
            self.centerline_corresponding_r[i] = r_closest_centerline
        
    def __len__(self):
        return self.ego_vehicle_positions.shape[0]-30