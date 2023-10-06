
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
from multiprocessing.shared_memory import SharedMemory

KEYS_WE_CARE_ABOUT : set = {
    "hist",
    "hist_quats",
    "hist_vel",
    "hist_spline_der",
    "fut",
    "fut_quats",
    "fut_tangents",
    "fut_vel",
    "fut_spline_der",
    "fut_speed",
    "future_arclength",
    "future_arclength_2d",
    "left_bd",
    "left_bd_tangents",
    "right_bd",
    "right_bd_tangents",

    "future_left_bd",
    "future_left_bd_tangents",
    
    "future_right_bd",
    "future_right_bd_tangents",
    
    "future_centerline",
    "future_centerline_tangents",

    "future_raceline",
    "future_raceline_tangents",
    
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
    def __init__(self):
        self.len : int = 0
        self.metadata : dict = dict()
        self.data_dict : dict[str,np.ndarray] = dict()
        self.subset_flag : deepracing_models.data_loading.SubsetFlag | None = None
        self.directory : str = None
        self.reference_curves : np.ndarray | None = None
        self.shared_mem_buffers = []
    @staticmethod
    def from_shared_memory(shared_memory_blocks : dict[str,tuple[str, list]], 
                           metadata : dict,
                           subset_flag : deepracing_models.data_loading.SubsetFlag,
                           dtype : np.dtype = np.float64) -> 'TrajectoryPredictionDataset':
        rtn : TrajectoryPredictionDataset = TrajectoryPredictionDataset()
        rtn.subset_flag = subset_flag
        rtn.metadata = metadata
        if subset_flag == deepracing_models.data_loading.SubsetFlag.TRAIN:
            rtn.len = rtn.metadata["num_train_samples"]
        elif subset_flag == deepracing_models.data_loading.SubsetFlag.VAL:
            rtn.len = rtn.metadata["num_val_samples"]
        elif subset_flag == deepracing_models.data_loading.SubsetFlag.TEST:
            rtn.len = rtn.metadata["num_test_samples"]
        elif subset_flag == deepracing_models.data_loading.SubsetFlag.ALL:
            rtn.len = rtn.metadata["num_samples"]
        else:
            raise ValueError("Invalid subset_flag: %d" % (subset_flag,))
        for k in KEYS_WE_CARE_ABOUT:
            name, shape = shared_memory_blocks[k]
            if not (shape[0]==rtn.len):
                raise ValueError("metadata indicates length %d, but shared memory block has first dimension %d" % (rtn.len, shape[0]))
            shared_mem : SharedMemory = SharedMemory(name=name)
            rtn.shared_mem_buffers.append(shared_mem)
            rtn.data_dict[k] = np.frombuffer(buffer=shared_mem.buf, dtype=dtype).reshape(shape)
        reference_curves_key = "reference_curves"
        if reference_curves_key in shared_memory_blocks.keys():
            name, shape = shared_memory_blocks[reference_curves_key]
            shared_mem : SharedMemory = SharedMemory(name=name)
            rtn.shared_mem_buffers.append(shared_mem)
            rtn.reference_curves = np.frombuffer(buffer=shared_mem.buf, dtype=dtype).reshape(shape)
        return rtn
    @staticmethod
    def from_file(metadatafile : str, subset_flag : deepracing_models.data_loading.SubsetFlag, dtype=np.float64) -> 'TrajectoryPredictionDataset':
        rtn : TrajectoryPredictionDataset = TrajectoryPredictionDataset()
        rtn.subset_flag = subset_flag
        with open(metadatafile, "r") as f:
            rtn.metadata = yaml.load(f, Loader=yaml.SafeLoader)
        rtn.directory = os.path.dirname(metadatafile)
        if subset_flag == deepracing_models.data_loading.SubsetFlag.TRAIN:
            npfile : str = os.path.join(rtn.directory, rtn.metadata["train_data"])
            rtn.len = rtn.metadata["num_train_samples"]
        elif subset_flag == deepracing_models.data_loading.SubsetFlag.VAL:
            npfile : str = os.path.join(rtn.directory, rtn.metadata["val_data"])
            rtn.len = rtn.metadata["num_val_samples"]
        elif subset_flag == deepracing_models.data_loading.SubsetFlag.TEST:
            npfile : str = os.path.join(rtn.directory, rtn.metadata["test_data"])
            rtn.len = rtn.metadata["num_test_samples"]
        elif subset_flag == deepracing_models.data_loading.SubsetFlag.ALL:
            npfile : str = os.path.join(rtn.directory, rtn.metadata["all_data"])
            rtn.len = rtn.metadata["num_samples"]
        else:
            raise ValueError("Invalid subset_flag: %d" % (subset_flag,))
        with open(npfile, "rb") as f:
            npdict : npio.NpzFile = np.load(f)
            for k in KEYS_WE_CARE_ABOUT:
                arr : np.ndarray = npdict[k]
                if not arr.shape[0] == rtn.len:
                    raise ValueError("Data key %s has length %d not consistent with metadata length %d" % (k, arr.shape[0], rtn.len))
                rtn.data_dict[k] = arr.copy().astype(dtype)
        return rtn
    def fit_bezier_curves(self, kbezier : int, device=torch.device("cpu"), cache=False):    
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
        all_deltar_flat = all_r_flat[:,-1] - all_r_flat[:,0]
        all_s_flat : torch.Tensor = (all_r_flat - all_r_flat[:,0,None])/(all_deltar_flat[:,None])

        left_bd = torch.as_tensor(self.data_dict["future_left_bd"], device=device)
        # left_bd_p0 = left_bd[:,0]
        # left_bd_t0 = torch.as_tensor(self.data_dict["future_left_bd_tangents"][:,0], device=device)

        right_bd = torch.as_tensor(self.data_dict["future_right_bd"], device=device)
        # right_bd_p0 = right_bd[:,0]
        # right_bd_t0 = torch.as_tensor(self.data_dict["future_right_bd_tangents"][:,0], device=device)

        centerline = torch.as_tensor(self.data_dict["future_centerline"], device=device)
        # centerline_p0 = centerline[:,0]
        # centerline_t0 = torch.as_tensor(self.data_dict["future_centerline_tangents"][:,0], device=device)

        raceline = torch.as_tensor(self.data_dict["future_raceline"], device=device)
        # raceline_p0 = raceline[:,0]
        # raceline_t0 = torch.as_tensor(self.data_dict["future_raceline_tangents"][:,0], device=device)

        all_lines : torch.Tensor = torch.stack([left_bd, right_bd, centerline, raceline], dim=1).to(device)
        # all_p0 : torch.Tensor = torch.stack([left_bd_p0, right_bd_p0, centerline_p0, raceline_p0], dim=1).to(device)
        # all_t0 : torch.Tensor = torch.stack([left_bd_t0, right_bd_t0, centerline_t0, raceline_t0], dim=1).to(device)

        all_lines_flat : torch.Tensor = all_lines.view(-1, all_lines.shape[-2], all_lines.shape[-1])
        # all_p0_flat : torch.Tensor = all_p0.view(-1, all_p0.shape[-1])
        # all_V0_flat : torch.Tensor = (all_t0.view(-1, all_t0.shape[-1]))*all_deltar_flat[:,None]
        
        print("Doing the lstsq fit, HERE WE GOOOOOO!", flush=True)
        _, all_curves_flat = deepracing_models.math_utils.bezierLsqfit(
            all_lines_flat, kbezier, t=all_s_flat#, P0=all_p0_flat, V0=all_V0_flat
            )
        all_curves_numpy : np.ndarray = \
            all_curves_flat.reshape(-1, 4, kbezier+1, all_lines.shape[-1]).cpu().numpy().astype(self.data_dict["future_left_bd_arclength"].dtype)
        npdict = {
            "left" : all_curves_numpy[:,0],
            "right" : all_curves_numpy[:,1],
            "center" : all_curves_numpy[:,2],
            "race" : all_curves_numpy[:,3]
        }
        # with open(cachefile, "wb") as f:
        #     np.savez_compressed(f, **npdict)
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
        