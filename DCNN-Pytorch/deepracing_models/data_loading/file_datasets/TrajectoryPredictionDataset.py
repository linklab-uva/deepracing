
import torch
import torch.utils
import torch.utils.data
import numpy as np, numpy.lib.npyio as npio
import deepracing_models
import deepracing_models.math_utils
import deepracing_models.data_loading
from tqdm import tqdm
import torch.jit
import os
import yaml
KEYS_WE_CARE_ABOUT : set= {
    "hist",
    "fut",
    "fut_tangents",
    "fut_speed",
    "future_arclength",
    "left_bd",
    "right_bd",
    "future_left_bd",
    "future_right_bd",
    "future_centerline",
    "future_raceline",
    "future_left_bd_arclength",
    "future_right_bd_arclength",
    "future_centerline_arclength",
    "future_raceline_arclength"
}
class TrajectoryPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, metadatafile : str, subset_flag : deepracing_models.data_loading.SubsetFlag,\
                  dtype=torch.float64, device=torch.device("cpu")):
        with open(metadatafile, "r") as f:
            self.metadata = yaml.load(f, Loader=yaml.SafeLoader)
        directory : str = os.path.dirname(metadatafile)
        if subset_flag == deepracing_models.data_loading.SubsetFlag.TRAIN:
            npfile : str = os.path.join(directory, self.metadata["train_data"])
            self.len = self.metadata["num_train_samples"]
        elif subset_flag == deepracing_models.data_loading.SubsetFlag.VAL:
            npfile : str = os.path.join(directory, self.metadata["val_data"])
            self.len = self.metadata["num_val_samples"]
        elif subset_flag == deepracing_models.data_loading.SubsetFlag.TEST:
            npfile : str = os.path.join(directory, self.metadata["test_data"])
            self.len = self.metadata["num_test_samples"]
        elif subset_flag == deepracing_models.data_loading.SubsetFlag.ALL:
            npfile : str = os.path.join(directory, self.metadata["all_data"])
            self.len = self.metadata["num_samples"]
        else:
            raise ValueError("Invalid subset_flag: %d" % (subset_flag,))

        self.data_dict : dict[str, torch.Tensor] = dict()
        print("Loading data at %s" % (npfile,), flush=True)
        with open(npfile, "rb") as f:
            npdict : npio.NpzFile = np.load(f)
            for k in KEYS_WE_CARE_ABOUT:
                arr : np.ndarray = npdict[k]
                if not arr.shape[0] == self.len:
                    raise ValueError("Data array %s has length %d not consistent with metadata length %d" % (k, arr.shape[0], self.len))
                self.data_dict[k] = torch.as_tensor(arr.copy(), dtype=dtype, device=device)
        self.directory : str = directory
        self.subset_flag : deepracing_models.data_loading.SubsetFlag = subset_flag
    def fit_bezier_curves(self, kbezier : int, device=torch.device("cpu")):    
               
        desc = "Fitting bezier curves for %s" % (self.directory,)
        print(desc, flush=True)

        left_bd_r : torch.Tensor = self.data_dict["future_left_bd_arclength"]
        right_bd_r : torch.Tensor = self.data_dict["future_right_bd_arclength"]
        centerline_r : torch.Tensor = self.data_dict["future_centerline_arclength"]
        raceline_r : torch.Tensor = self.data_dict["future_raceline_arclength"]
        
        all_r : torch.Tensor = torch.stack([left_bd_r, right_bd_r, centerline_r, raceline_r], dim=1).to(device)
        all_r_flat : torch.Tensor = all_r.view(-1, all_r.shape[-1])
        all_s_flat : torch.Tensor = (all_r_flat - all_r_flat[:,0,None])/((all_r_flat[:,-1] - all_r_flat[:,0])[:,None])

        left_bd : torch.Tensor = self.data_dict["future_left_bd"]
        right_bd : torch.Tensor = self.data_dict["future_right_bd"]
        centerline : torch.Tensor = self.data_dict["future_centerline"]
        raceline : torch.Tensor = self.data_dict["future_raceline"]

        all_lines : torch.Tensor = torch.stack([left_bd, right_bd, centerline, raceline], dim=1).to(device)
        all_lines_flat : torch.Tensor = all_lines.view(-1, all_lines.shape[-2], all_lines.shape[-1])
        
        print("Doing the lstsq fit, HERE WE GOOOOOO!", flush=True)
        _, all_curves_flat = deepracing_models.math_utils.bezierLsqfit(all_lines_flat, kbezier, t=all_s_flat)
        self.data_dict["reference_curves"] = all_curves_flat.to(self.data_dict["reference_curves"].device).reshape(-1, 4, kbezier+1, all_lines.shape[-1])
        print("Done", flush=True)


    def __len__(self):
        return self.len
    def __getitem__(self, index):
        return {k : self.data_dict[k][index] for k in self.data_dict.keys()}
        