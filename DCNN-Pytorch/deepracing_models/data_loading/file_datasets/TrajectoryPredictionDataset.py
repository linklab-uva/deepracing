
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

class TrajectoryPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, metadatafile : str, subset_flag : deepracing_models.data_loading.SubsetFlag,\
                  kbezier=3, dtype=torch.float64, device=torch.device("cpu")):
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
        with open(npfile, "rb") as f:
            npdict : npio.NpzFile = np.load(f)
            for k in npdict.keys():
                arr : np.ndarray = npdict[k]
                if not arr.shape[0] == self.len:
                    raise ValueError("Data array %s has length %d not consistent with metadata length %d" % (k, arr.shape[0], self.len))
                self.data_dict[k] = torch.as_tensor(arr.copy(), dtype=dtype, device=device)
        
        self.data_dict["reference_curves"] = torch.empty([self.len, 4, kbezier+1, 3], dtype=dtype, device=device)

        for i in tqdm(range(self.len), desc="Fitting bezier curves"):

            left_bd_r : torch.Tensor = self.data_dict["future_left_bd_arclength"][i]
            left_bd_s = (left_bd_r - left_bd_r[0])/(left_bd_r[-1] - left_bd_r[0])
            self.data_dict["future_left_bd_arclength"][i] = left_bd_r - left_bd_r[0]

            right_bd_r : torch.Tensor = self.data_dict["future_right_bd_arclength"][i]
            right_bd_s = (right_bd_r - right_bd_r[0])/(right_bd_r[-1] - right_bd_r[0])
            self.data_dict["future_right_bd_arclength"][i] = right_bd_r - right_bd_r[0]

            centerline_r : torch.Tensor = self.data_dict["future_centerline_arclength"][i]
            centerline_s = (centerline_r - centerline_r[0])/(centerline_r[-1] - centerline_r[0])
            self.data_dict["future_centerline_arclength"][i] = centerline_r - centerline_r[0]

            raceline_r : torch.Tensor = self.data_dict["future_raceline_arclength"][i]
            raceline_s = (raceline_r - raceline_r[0])/(raceline_r[-1] - raceline_r[0])
            self.data_dict["future_raceline_arclength"][i] = raceline_r - raceline_r[0]


            left_bd : torch.Tensor = self.data_dict["future_left_bd"][i]
            right_bd : torch.Tensor = self.data_dict["future_right_bd"][i]
            centerline : torch.Tensor = self.data_dict["future_centerline"][i]
            raceline : torch.Tensor = self.data_dict["future_raceline"][i]
            all_lines : torch.Tensor = torch.stack([left_bd, right_bd, centerline, raceline], dim=0)

            all_s : torch.Tensor = torch.stack([left_bd_s, right_bd_s, centerline_s, raceline_s], dim=0)
 
            self.data_dict["reference_curves"][i] = deepracing_models.math_utils.bezierLsqfit(all_lines, kbezier, t=all_s)[1]

    def __len__(self):
        return self.len
    def __getitem__(self, index):
        return {k : self.data_dict[k][index] for k in self.data_dict.keys()}
        