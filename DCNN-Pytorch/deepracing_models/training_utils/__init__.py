import comet_ml
from deepracing_models.data_loading import file_datasets as FD, SubsetFlag 
from deepracing_models.data_loading.utils.file_utils import load_datasets_from_files
from deepracing_models.nn_models.trajectory_prediction import BARTE
import torch, torch.nn, torch.utils.data as torchdata
import yaml
import os
import numpy as np
import traceback
import sys
import deepracing_models.math_utils
import tqdm
from contextlib import nullcontext

class BarteTrainer:
    def __init__(self, config_file : str, gpu=0) -> None:
        self.config_file = config_file
        with open(config_file, "r") as f:
            self.config : dict = yaml.safe_load(f)
        api_key=os.getenv("COMET_API_KEY")
        if (api_key is not None) and len(api_key)>0:
            self.comet_experiment = comet_ml.Experiment(api_key=api_key, 
                                                project_name="bamf", 
                                                workspace="electric-turtle",
                                                auto_metric_logging=False
                                                )
            self.comet_experiment.log_asset(config_file, file_name="config.yaml", overwrite=True)
        else:
            self.comet_experiment = None
        self.comet_config : dict = self.config["comet"]
        self.trainerconfig : dict = self.config["trainer"]   
        self.netconfig : dict = self.config["network"]
        self.dataconfig : dict = self.config["data"]
        self.training_sets : list[FD.TrajectoryPredictionDataset] | None = None
        self.validation_sets : list[FD.TrajectoryPredictionDataset] | None = None
        num_segments = self.netconfig["num_segments"]
        kbezier = self.netconfig["kbezier"]
        with_batchnorm = self.netconfig["with_batchnorm"]
        self.heading_encoding = self.netconfig.get("heading_encoding", "quaternion")
        if self.heading_encoding=="angle":
            print("Using heading angle as orientation input")
            history_dimension = 5
        elif self.heading_encoding=="quaternion":
            print("Using quaternion as orientation input")
            history_dimension = 6
        else:
            raise ValueError("Unknown heading encoding: %s" % (self.heading_encoding,))
        self.network : BARTE = BARTE( history_dimension = history_dimension,
                num_segments = num_segments, 
                kbezier = kbezier,
                with_batchnorm = with_batchnorm
            )
        self.trainerconfig : dict = self.config["trainer"]
        if self.trainerconfig["float32"]:
            self.network = self.network.float()
        else:
            self.network = self.network.double()
        if gpu>=0:
            self.network = self.network.cuda(gpu)
        self.loss_function = self.trainerconfig.get("loss_function", "ade")
        self.loss_weights = self.trainerconfig.get("loss_weights")
        if not self.loss_function in {"ade", "lat_long"}:
            raise ValueError("Loss function must one of: lat_long, ade")
        if self.loss_function=="lat_long" and (self.loss_weights is None):
            raise ValueError("loss_weights must be specified when using lat_long loss")
        self.optimizer : torch.optim.Adam | None = None
        self.epoch_number : int | None = None
    def prepare_for_training(self):
        lr = float(self.trainerconfig["learning_rate"])
        betas = tuple(self.trainerconfig["betas"])
        weight_decay = self.trainerconfig["weight_decay"]
        self.network = self.network.train()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = lr, betas = betas, weight_decay = weight_decay)
        self.epoch_number = 1
    def init_training_data(self, keys : set):
        search_dirs = self.dataconfig["dirs"]
        self.training_sets = []
        for search_dir in search_dirs:
            self.training_sets += load_datasets_from_files(search_dir, keys=keys, dtype=np.float64)
    def init_validation_data(self, keys : set):
        search_dirs = self.dataconfig["dirs"]
        self.validation_sets = []
        for search_dir in search_dirs:
            self.validation_sets += load_datasets_from_files(search_dir, keys=keys, flag=SubsetFlag.VAL, dtype=np.float64)
    def forward_pass(self, datadict : dict[str,torch.Tensor], 
                     tstart : torch.Tensor,
                     dt : torch.Tensor,
                     tsamp : torch.Tensor,
                     coordinate_idx_history=[0,1], 
                     quaternion_idx_history=[2,3]):
        dtype=tstart.dtype
        device=tstart.device
        up = torch.as_tensor([0.0, 0.0, 1.0], dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        position_history = datadict["hist"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
        vel_history = datadict["hist_vel"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
        quat_history = datadict["hist_quats"][:,:,quaternion_idx_history].to(device=device, dtype=dtype)
        quat_history = quat_history*(torch.sign(quat_history[:,:,-1])[...,None])
        quat_history = quat_history/torch.norm(quat_history, p=2.0, dim=-1, keepdim=True)
        if self.heading_encoding=="quaternion":
            quat_input = quat_history
        elif self.heading_encoding=="angle":
            qz = quat_history[:,:,-2]
            qw = quat_history[:,:,-1]
            quat_input = 2.0*torch.atan2(qz,qw).unsqueeze(-1)


        position_future = datadict["fut"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
        vel_future = datadict["fut_vel"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
        fut_tangents_full = datadict["fut_tangents"].to(device=device, dtype=dtype)
        fut_normals_full = torch.cross(up.expand_as(fut_tangents_full), fut_tangents_full, dim=-1)
        fut_normals = fut_normals_full[:,:,coordinate_idx_history]
        fut_normals /= torch.norm(fut_normals, p=2.0, dim=-1, keepdim=True)
        fut_tangents = fut_tangents_full[:,:,coordinate_idx_history]
        fut_tangents /= torch.norm(fut_tangents, p=2.0, dim=-1, keepdim=True)
        _2d = len(coordinate_idx_history)==2
        if _2d:
            rotmats = torch.stack([fut_tangents, fut_normals], dim=-1)
        else:
            zvecs = torch.cross(fut_tangents, fut_normals, dim=-1)
            zvecs /= torch.norm(zvecs, p=2.0, dim=-1, keepdim=True)
            rotmats = torch.stack([fut_tangents, fut_normals, zvecs], dim=-1)
        Rtransform = rotmats.mT
        Ptransform = torch.matmul(Rtransform, -position_future.unsqueeze(-1)).squeeze(-1)


        left_bound_input = datadict["left_bd"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
        right_bound_input = datadict["right_bd"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
        left_bound_tangents = datadict["left_bd_tangents"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
        right_bound_tangents = datadict["right_bd_tangents"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
        left_bound_tangents = left_bound_tangents/torch.norm(left_bound_tangents, p=2.0, dim=-1, keepdim=True)
        right_bound_tangents = right_bound_tangents/torch.norm(right_bound_tangents, p=2.0, dim=-1, keepdim=True)

        history_inputs = torch.cat([position_history, vel_history, quat_input], dim=-1)
        left_boundary_inputs = torch.cat([left_bound_input, left_bound_tangents], dim=-1)
        right_boundary_inputs = torch.cat([right_bound_input, right_bound_tangents], dim=-1)

        p0 = position_future[:,0]
        v0 = vel_future[:,0]
        _, poscurveout = self.network(history_inputs, left_boundary_inputs, right_boundary_inputs, dt, v0, p0=p0)
        pout, _ = deepracing_models.math_utils.compositeBezierEval(tstart, dt, poscurveout, tsamp)
        deltas = torch.matmul(Rtransform, pout.unsqueeze(-1)).squeeze(-1) + Ptransform
        return pout, deltas
    
    def run_epoch(self, epoch_number, train=True, workers=0):
        comet_experiment = self.comet_experiment
        if train:
            datasets = self.training_sets
            self.network = self.network.requires_grad_(requires_grad=True).train()
            if comet_experiment is not None:
                comet_experiment.train()
        else:
            datasets = self.validation_sets
            self.network = self.network.requires_grad_(requires_grad=False).eval()
        batch_size = self.trainerconfig["batch_size"]
        concat_dataset : torchdata.ConcatDataset = torchdata.ConcatDataset(datasets)
        dataloader : torchdata.DataLoader = torchdata.DataLoader(concat_dataset, batch_size=batch_size, pin_memory=True, shuffle=train, num_workers=workers)
        dataloader_enumerate = enumerate(dataloader)

        if comet_experiment is None:
            tq : tqdm.tqdm  = tqdm.tqdm(dataloader_enumerate, desc="Yay")
        else:
            tq : enumerate  = dataloader_enumerate
        print("Running epoch %d" % (self.epoch_number,), flush=True)
        if comet_experiment is not None:
            comet_experiment.set_epoch(epoch_number)   
        firstparam = next(self.network.parameters())
        dtype = firstparam.dtype
        device = firstparam.device
        Nfuture = datasets[0].metadata["numsamples_prediction"]
        tfuture = datasets[0].metadata["predictiontime"]
        num_segments = self.netconfig["num_segments"]
        with torch.no_grad() if (not train) else nullcontext() as f:
            tsegs : torch.Tensor = torch.linspace(0.0, tfuture, steps=num_segments+1, device=device, dtype=dtype)
            tstart_ = tsegs[:-1]
            dt_ = tsegs[1:] - tstart_
            tsamp_ : torch.Tensor = torch.linspace(0.0, tfuture, steps=Nfuture, device=device, dtype=dtype)
            for (i, dict_) in tq:
                datadict : dict[str,torch.Tensor] = dict_
                currentbatchdim = datadict["hist"].shape[0]
                tstart = tstart_[None].expand(currentbatchdim, num_segments)
                dt = dt_[None].expand(currentbatchdim, num_segments)
                tsamp = tsamp_[None].expand(currentbatchdim, Nfuture)
                pout, deltas = self.forward_pass(datadict, tstart, dt, tsamp)

                longitudinal_error = torch.mean(torch.square(deltas[:,:,0]))
                lateral_error = torch.mean(torch.square(deltas[:,:,1]))
                ade = torch.mean(torch.square(deltas))
                if train:
                    if self.loss_function=="ade":
                        loss = ade
                    elif self.loss_function=="lat_long":
                        loss = self.loss_weights[0]*lateral_error + self.loss_weights[1]*longitudinal_error
                    else:
                        raise ValueError("Invalid loss function: %s" % (self.loss_function,))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                if (i%10)==0:
                    metric_dict = {
                            "ade" : ade.item(),
                            "lateral_error" : lateral_error.item(),
                            "longitudinal_error" : longitudinal_error.item(),
                            "loss" : loss.item(),
                    }
                    comet_experiment.log_metrics(metric_dict)
            