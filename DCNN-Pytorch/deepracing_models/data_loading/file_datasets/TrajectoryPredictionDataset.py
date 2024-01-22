
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
import scipy.interpolate

class TrajectoryPredictionDataset(torch.utils.data.Dataset):
    ALL_KEYS : set = { 
        "thistory",
        "tfuture",
        "current_position",
        "current_orientation",
        "hist",
        "hist_quats",
        "hist_tangents",
        "hist_spline_der",
        "hist_vel",
        "hist_accel",
        "hist_speed",
        "hist_angvel",
        "fut",
        "fut_quats",
        "fut_tangents",
        "fut_spline_der",
        "fut_vel",
        "fut_accel",
        "fut_speed",
        "fut_angvel",
        "left_bd",
        "left_bd_tangents",
        "right_bd",
        "right_bd_tangents",
        "future_arclength",
        "future_arclength_2d",
        "future_arclengths_fromspline",
        "future_arclengths_fromspline_2d",
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
        "future_raceline_arclength"
    }
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
    def __init__(self):
        self.len : int = 0
        self.metadata : dict = dict()
        self.data_dict : dict[str,np.ndarray] = dict()
        self.subset_flag : deepracing_models.data_loading.SubsetFlag | None = None
        self.directory : str = None
        self.reference_curves : np.ndarray | None = None
        self.reference_curves_rswitch : np.ndarray | None = None
        self.shared_mem_buffers = []
        self.mtr_polyline_config : dict | None = None
    @staticmethod
    def from_shared_memory(shared_memory_blocks : dict[str,tuple[str, list]], 
                           metadata : dict,
                           subset_flag : deepracing_models.data_loading.SubsetFlag, 
                           keys=KEYS_WE_CARE_ABOUT,
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
        for k in keys:
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
    def from_file(metadatafile : str, subset_flag : deepracing_models.data_loading.SubsetFlag, keys=KEYS_WE_CARE_ABOUT, dtype=np.float64) -> 'TrajectoryPredictionDataset':
        print("Loading data for %s" % (metadatafile,))
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
            # print(list(npdict.keys()))
            for k in keys:
                try:
                    arr : np.ndarray = npdict[k]
                except KeyError as e:
                    if k in {"left_bd_tangents", "right_bd_tangents"}:
                        continue
                    else:
                        raise e
                if not arr.shape[0] == rtn.len:
                    raise ValueError("Data key %s has length %d not consistent with metadata length %d" % (k, arr.shape[0], rtn.len))
                rtn.data_dict[k] = arr.copy().astype(dtype)
        if ("left_bd_tangents" in keys) and not ("left_bd_tangents" in rtn.data_dict.keys()):
            rtn.compute_tangents("left_bd", "left_bd_tangents")
        if ("right_bd_tangents" in keys) and not ("right_bd_tangents" in rtn.data_dict.keys()):
            rtn.compute_tangents("right_bd", "right_bd_tangents")
        return rtn
    def compute_tangents(self, key : str, key_out : str):
        print("Computing tangents for key: %s" % (key,))
        points : np.ndarray = self.data_dict[key]
        self.data_dict[key_out] = np.zeros_like(points)
        for idx in range(points.shape[0]):
            current_points = points[idx]
            euclidean_deltas = np.zeros_like(current_points[:,0])
            euclidean_deltas[1:]=np.cumsum(np.linalg.norm(current_points[1:] - current_points[:-1], ord=2.0, axis=1))
            spl : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(euclidean_deltas, current_points, bc_type="natural")
            current_tangents = spl(euclidean_deltas, nu=1)
            self.data_dict[key_out][idx] = (current_tangents/np.linalg.norm(current_tangents, ord=2.0, axis=1, keepdims=True))[:]
    def fit_bezier_curves(self, kbezier : int, device=torch.device("cpu"), cache=False, segments = 1, constrain_tangents = True):    
        cachefile = os.path.join(self.directory, "bcurve_order_%d_segments_%d.npz" % (kbezier, segments))
        if cache and os.path.isfile(cachefile):
            desc = "Loading bezier curves from cache file %s" % (cachefile,)
            print(desc, flush=True)
            with open(cachefile, "rb") as f:
                npdict : npio.NpzFile = np.load(f)
                self.reference_curves = npdict["all_curves"].copy()
                if "all_rswitch" in npdict.keys():
                    self.reference_curves_rswitch = npdict["all_rswitch"].copy()
            return
        desc = "Fitting bezier curves for %s" % (self.directory,)
        print(desc, flush=True)

        left_bd_r = torch.as_tensor(self.data_dict["future_left_bd_arclength"], device=device)
        right_bd_r = torch.as_tensor(self.data_dict["future_right_bd_arclength"], device=device)
        centerline_r = torch.as_tensor(self.data_dict["future_centerline_arclength"], device=device)
        raceline_r = torch.as_tensor(self.data_dict["future_raceline_arclength"], device=device)
        
        all_r = torch.stack([
            left_bd_r, 
            right_bd_r, 
            centerline_r, 
            raceline_r], dim=1).to(device)
        all_r = all_r - all_r[:,:,0,None]
        all_r_flat = all_r.view(-1, all_r.shape[-1])
        
        left_bd = torch.as_tensor(self.data_dict["future_left_bd"], device=device)
        right_bd = torch.as_tensor(self.data_dict["future_right_bd"], device=device)
        centerline = torch.as_tensor(self.data_dict["future_centerline"], device=device)
        raceline = torch.as_tensor(self.data_dict["future_raceline"], device=device)

        all_lines : torch.Tensor = torch.stack([left_bd, right_bd, centerline, raceline], dim=1).to(device)
        all_lines_flat : torch.Tensor = all_lines.view(-1, all_lines.shape[-2], all_lines.shape[-1]) 

        print("Doing the lstsq fit, HERE WE GOOOOOO!", flush=True)
        if segments==1:
            all_s_flat : torch.Tensor = all_r_flat/all_r_flat[:,-1,None]
            _, all_curves_flat = deepracing_models.math_utils.bezierLsqfit(
                all_lines_flat, kbezier, t=all_s_flat#, P0=all_p0_flat, V0=all_V0_flat
                )
            self.reference_curves : np.ndarray = all_curves_flat.view(all_r.shape[0], 4, kbezier+1, -1).cpu().numpy().astype(self.data_dict["future_left_bd_arclength"].dtype)
        else:
            dYdT_0, dYdT_f = None, None
            if constrain_tangents:
                left_bd_t0 = torch.as_tensor(self.data_dict["future_left_bd_tangents"][:,0], device=device)
                right_bd_t0 = torch.as_tensor(self.data_dict["future_right_bd_tangents"][:,0], device=device)
                centerline_t0 = torch.as_tensor(self.data_dict["future_centerline_tangents"][:,0], device=device)
                raceline_t0 = torch.as_tensor(self.data_dict["future_raceline_tangents"][:,0], device=device)
                dYdT_0 = torch.stack([left_bd_t0, right_bd_t0, centerline_t0, raceline_t0], dim=1).to(device)

                left_bd_tf = torch.as_tensor(self.data_dict["future_left_bd_tangents"][:,-1], device=device)
                right_bd_tf = torch.as_tensor(self.data_dict["future_right_bd_tangents"][:,-1], device=device)
                centerline_tf = torch.as_tensor(self.data_dict["future_centerline_tangents"][:,-1], device=device)
                raceline_tf = torch.as_tensor(self.data_dict["future_raceline_tangents"][:,-1], device=device)
                dYdT_f = torch.stack([left_bd_tf, right_bd_tf, centerline_tf, raceline_tf], dim=1).to(device)

            all_curves, all_rswitch = deepracing_models.math_utils.compositeBezierFit(
                all_r, all_lines, segments, kbezier=kbezier, dYdT_0=dYdT_0, dYdT_f=dYdT_f
            )
            self.reference_curves_rswitch = all_rswitch.cpu().numpy().astype(self.data_dict["future_left_bd_arclength"].dtype).copy()
            self.reference_curves : np.ndarray = all_curves.cpu().numpy().astype(self.data_dict["future_left_bd_arclength"].dtype).copy()
        datadump : dict = {"all_curves" : self.reference_curves}
        if segments>1:
            datadump["all_rswitch"] = self.reference_curves_rswitch
        with open(cachefile, "wb") as f:
            np.savez(f, **datadump)
        print("Done", flush=True)

    def __len__(self):
        return self.len
    def __getitem__(self, index):
        datadict = {k : self.data_dict[k][index] for k in self.data_dict.keys()}
        datadict["trackname"] = self.metadata["trackname"]
        if self.reference_curves is not None:
            datadict["reference_curves"] = self.reference_curves[index]
        if self.reference_curves_rswitch is not None:
            datadict["reference_curves_rswitch"] = self.reference_curves_rswitch[index]
        if self.mtr_polyline_config is not None:
            import deepracing_models.data_loading.utils.mtr_utils as mtr_utils
            scene_id = "%s_%d" % (self.directory, index)
            return mtr_utils.deepracing_to_mtr(datadict, scene_id, self.mtr_polyline_config)
        else:
            return datadict
        