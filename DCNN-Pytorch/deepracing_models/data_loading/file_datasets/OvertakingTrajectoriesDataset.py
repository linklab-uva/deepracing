import torch
from torch.utils.data import Dataset
import numpy as np

class OvertakingTrajectoriesDataset(Dataset):
    def __init__(self, data_dict : dict[str, np.ndarray], metadata_dict : dict):
        self.metadata_dict = metadata_dict
        self.data_dict = {k : v.copy() for k,v in data_dict.items()}

    def __len__(self):
        return self.data_dict["attacker_pos"].shape[0]

    def __getitem__(self, idx):
        
        rtn = {k : v[idx] for k,v in self.data_dict.items() if k not in {"tcurrent","delta_t"}}
        rtn["tcurrent"] = self.data_dict["tcurrent"][idx]
        rtn["delta_t"] = self.data_dict["delta_t"]
        rtn["track_name"] = self.metadata_dict["track_name"]

        return rtn