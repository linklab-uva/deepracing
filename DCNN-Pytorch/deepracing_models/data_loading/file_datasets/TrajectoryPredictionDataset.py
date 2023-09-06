
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
import io

KEYS_WE_CARE_ABOUT : set = {
    "hist",
    "fut",
    "fut_tangents",
    "fut_vel",
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
    "future_raceline_arclength",
    "thistory",
    "tfuture",
    "current_position",
    "current_orientation" 
}
class TrajectoryPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, metadatafile : str, subset_flag : deepracing_models.data_loading.SubsetFlag, direct_load : bool,\
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

        print("Loading data at %s" % (npfile,), flush=True)
        self.data_dict : npio.NpzFile | dict[str, torch.Tensor] = None
        if direct_load:
            self.file_handle = None
            self.data_dict = dict()
            with open(npfile, "rb") as f:
                npdict : npio.NpzFile = np.load(f)
                for k in KEYS_WE_CARE_ABOUT:
                    arr : np.ndarray = npdict[k]
                    if not arr.shape[0] == self.len:
                        raise ValueError("Data key %s has length %d not consistent with metadata length %d" % (k, arr.shape[0], self.len))
                    self.data_dict[k] = arr.copy()
        else:
            self.file_handle = open(npfile, "rb")
            self.data_dict : npio.NpzFile = np.load(self.file_handle)
        self.directory : str = directory
        self.subset_flag : deepracing_models.data_loading.SubsetFlag = subset_flag
        self.reference_curves : np.ndarray | None = None
        self.direct_load : bool = direct_load
    def __del__(self):
        if self.file_handle is not None:
            self.file_handle.close()
    def fit_bezier_curves(self, kbezier : int, device=torch.device("cpu"), built_in_lstq=False, cache=False):    
        cachefile = os.path.join(self.directory, "bcurve_order_%d.npz" % (kbezier,))
        if cache and os.path.isfile(cachefile):
            desc = "Loading bezier curves from cache file %s" % (cachefile,)
            print(desc, flush=True)
            with open(cachefile, "rb") as f:
                npdict : npio.NpzFile = np.load(f)
                self.reference_curves = np.stack([npdict[k].copy() for k in ["left", "right", "center", "race"]], axis=1)
            return
        desc = "Fitting bezier curves for %s" % (self.directory,)
        print(desc, flush=True)

        left_bd_r = torch.as_tensor(self.data_dict["future_left_bd_arclength"], device=device)
        right_bd_r = torch.as_tensor(self.data_dict["future_right_bd_arclength"], device=device)
        centerline_r = torch.as_tensor(self.data_dict["future_centerline_arclength"], device=device)
        raceline_r = torch.as_tensor(self.data_dict["future_raceline_arclength"], device=device)
        
        all_r = torch.stack([left_bd_r, right_bd_r, centerline_r, raceline_r], dim=1).to(device)
        all_r_flat = all_r.view(-1, all_r.shape[-1])
        all_s_flat : torch.Tensor = (all_r_flat - all_r_flat[:,0,None])/((all_r_flat[:,-1] - all_r_flat[:,0])[:,None])

        left_bd = torch.as_tensor(self.data_dict["future_left_bd"], device=device)
        right_bd = torch.as_tensor(self.data_dict["future_right_bd"], device=device)
        centerline = torch.as_tensor(self.data_dict["future_centerline"], device=device)
        raceline = torch.as_tensor(self.data_dict["future_raceline"], device=device)

        all_lines : torch.Tensor = torch.stack([left_bd, right_bd, centerline, raceline], dim=1).to(device)
        all_lines_flat : torch.Tensor = all_lines.view(-1, all_lines.shape[-2], all_lines.shape[-1])
        
        print("Doing the lstsq fit, HERE WE GOOOOOO!", flush=True)
        _, all_curves_flat = deepracing_models.math_utils.bezierLsqfit(all_lines_flat, kbezier, t=all_s_flat, built_in_lstq=built_in_lstq)
        all_curves_numpy : np.ndarray = all_curves_flat.reshape(-1, 4, kbezier+1, all_lines.shape[-1]).cpu().numpy()
        npdict = {
            "left" : all_curves_numpy[:,0],
            "right" : all_curves_numpy[:,1],
            "center" : all_curves_numpy[:,2],
            "race" : all_curves_numpy[:,3]
        }
        with open(cachefile, "wb") as f:
            np.savez_compressed(f, **npdict)
        self.reference_curves = all_curves_numpy.copy()
        print("Done", flush=True)


    def __len__(self):
        return self.len
    def __getitem__(self, index):
        datadict = {k : self.data_dict[k][index] for k in KEYS_WE_CARE_ABOUT}
        datadict["trackname"] = self.metadata["trackname"]
        if self.reference_curves is not None:
            datadict["reference_curves"] = self.reference_curves[index]
        return datadict
        