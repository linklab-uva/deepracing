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
from contextlib import nullcontext, ExitStack
import shutil

class BarteTrainer:
    def __init__(self, config : dict, scratch_dir : str, gpu=0, no_comet = False) -> None:
        self.config : dict = config
        api_key=os.getenv("COMET_API_KEY")
        if (api_key is None):
            raise ValueError("COMET_API_KEY environment variable must be set")
        self.comet_experiment = comet_ml.Experiment(api_key=api_key, 
                                            project_name="bamf", 
                                            workspace="electric-turtle",
                                            auto_metric_logging=False,
                                            disabled = no_comet
                                            )
        if no_comet:
            self.scratch_dir = os.path.join(scratch_dir, "debug")
        else:
            self.scratch_dir = os.path.join(scratch_dir, self.comet_experiment.get_name())
        if os.path.isdir(self.scratch_dir):
            shutil.rmtree(self.scratch_dir)
        os.makedirs(self.scratch_dir)
        cfg_file_out = os.path.join(self.scratch_dir, "config.yaml")
        with open(cfg_file_out, "w") as f:
            yaml.safe_dump(self.config, f, indent=3)
        self.comet_experiment.log_asset(cfg_file_out, file_name="config.yaml", overwrite=True)
        self.comet_config : dict = self.config["comet"]
        self.trainerconfig : dict = self.config["trainer"]   
        self.netconfig : dict = self.config["network"]
        self.dataconfig : dict = self.config["data"]
        self.comet_experiment.log_parameters(self.dataconfig)
        self.comet_experiment.log_parameters(self.netconfig)
        self.comet_experiment.log_parameters(self.trainerconfig)
        tags = self.comet_config.get("tags")
        if tags is not None:
            self.comet_experiment.add_tags(tags)
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
        self.loss_function = self.trainerconfig.get("loss_function", "squared_norm")
        self.loss_weights = self.trainerconfig.get("loss_weights")
        if not self.loss_function in {"squared_norm", "lat_long"}:
            raise ValueError("Loss function must one of: lat_long, squared_norm")
        if self.loss_function=="lat_long" and (self.loss_weights is None):
            raise ValueError("loss_weights must be specified when using lat_long loss")
        self.optimizer : torch.optim.Adam | None = None
    def build_optimizer(self):
        lr = float(self.trainerconfig["learning_rate"])
        betas = tuple(self.trainerconfig["betas"])
        weight_decay = self.trainerconfig["weight_decay"]
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = lr, betas = betas, weight_decay = weight_decay)
    def init_training_data(self, keys : set):
        print("Loading training data")
        search_dirs = self.dataconfig["dirs"]
        self.training_sets = []
        for search_dir in search_dirs:
            self.training_sets += load_datasets_from_files(search_dir, keys=keys, dtype=np.float64)
        rforward = self.training_sets[0].metadata["rforward"]
        predictiontime = self.training_sets[0].metadata["predictiontime"]
        historytime = self.training_sets[0].metadata["historytime"]
        self.comet_experiment.log_parameters({
            "rforward": rforward,
            "predictiontime": predictiontime,
            "historytime": historytime
        })
        if historytime!=3.0:
            self.comet_experiment.add_tag("history_time_ablation")
            self.comet_experiment.add_tag("%1.1fsecond_history" % (historytime,))
        if rforward!=400.0:
            self.comet_experiment.add_tag("boundary_length_ablation")
            self.comet_experiment.add_tag("%3.1fmeter_boundary_inputs" % (rforward,))
        if historytime==3.0 and rforward==400.0:
            self.comet_experiment.add_tag("standard_data")
    def init_validation_data(self, keys : set):
        print("Loading validation data")
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
    def checkpoint(self, epoch_number):
        network_file = os.path.join(self.scratch_dir, "model_%d.pt" % (epoch_number))
        with open(network_file, "wb") as f:
            torch.save(self.network.state_dict(), f)
        optimizer_file = os.path.join(self.scratch_dir, "optimizer_%d.pt" % (epoch_number))
        with open(optimizer_file, "wb") as f:
            torch.save(self.optimizer.state_dict(), f)
        self.comet_experiment.log_asset(network_file, file_name=os.path.basename(network_file), copy_to_tmp=False)  
        self.comet_experiment.log_asset(optimizer_file, file_name=os.path.basename(optimizer_file), copy_to_tmp=False)  
            
    def run_epoch(self, epoch_number, train=True, workers=0, with_tqdm=False):
        comet_experiment = self.comet_experiment
        comet_experiment.set_epoch(epoch_number)   
        if train:
            datasets = self.training_sets
            self.network = self.network.train()
        else:
            datasets = self.validation_sets
            self.network = self.network.eval()
        batch_size = self.trainerconfig["batch_size"]
        concat_dataset : torchdata.ConcatDataset = torchdata.ConcatDataset(datasets)
        dataloader : torchdata.DataLoader = torchdata.DataLoader(concat_dataset, batch_size=batch_size, pin_memory=True, shuffle=train, num_workers=workers)
        dataloader_enumerate = enumerate(dataloader)
        if with_tqdm:
            if train:
                prefix = "Training epoch %d" % (epoch_number,)
            else:
                prefix = "Testing epoch %d" % (epoch_number,)
            tq : tqdm.tqdm  = tqdm.tqdm(dataloader_enumerate, desc=prefix, total=int(np.ceil(len(concat_dataset)/batch_size)))
        else:
            tq : enumerate  = dataloader_enumerate
        firstparam = next(self.network.parameters())
        dtype = firstparam.dtype
        device = firstparam.device
        Nfuture = datasets[0].metadata["numsamples_prediction"]
        tfuture = datasets[0].metadata["predictiontime"]
        num_segments = self.netconfig["num_segments"]
        with ExitStack() as ctx:
            if train:
                comet_ctx = ctx.enter_context(comet_experiment.train())
            else:
                comet_ctx = ctx.enter_context(comet_experiment.test())
                no_grad = ctx.enter_context(torch.no_grad())
            tsegs : torch.Tensor = torch.linspace(0.0, tfuture, steps=num_segments+1, device=device, dtype=dtype)
            tstart_ = tsegs[:-1]
            dt_ = tsegs[1:] - tstart_
            tsamp_ : torch.Tensor = torch.linspace(0.0, tfuture, steps=Nfuture, device=device, dtype=dtype)
            batch_means = {
                "ade" : [],
                "lateral_error" : [],
                "longitudinal_error" : []
            }
            for (i, dict_) in tq:
                datadict : dict[str,torch.Tensor] = dict_
                currentbatchdim = datadict["hist"].shape[0]
                tstart = tstart_[None].expand(currentbatchdim, num_segments)
                dt = dt_[None].expand(currentbatchdim, num_segments)
                tsamp = tsamp_[None].expand(currentbatchdim, Nfuture)
                pout, deltas = self.forward_pass(datadict, tstart, dt, tsamp)

                longitudinal_error = torch.mean(torch.square(deltas[:,:,0]))
                lateral_error = torch.mean(torch.square(deltas[:,:,1]))
                ade = torch.mean(torch.norm(deltas, p=2.0, dim=-1))
                squared_norm = torch.mean(torch.square(deltas))
                if train:
                    if self.loss_function=="squared_norm":
                        loss = squared_norm
                    elif self.loss_function=="lat_long":
                        loss = self.loss_weights[0]*lateral_error + self.loss_weights[1]*longitudinal_error
                    else:
                        raise ValueError("Invalid loss function: %s" % (self.loss_function,))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                metric_dict = {
                    "ade" : ade.item(),
                    "squared_norm" : squared_norm.item(),
                    "lateral_error" : lateral_error.item(),
                    "longitudinal_error" : longitudinal_error.item()
                }
                if train:
                    metric_dict["loss"] = loss.item()
                    
                for k in batch_means.keys():
                    batch_means[k].append(metric_dict[k])
                if (i%10)==0:
                    comet_experiment.log_metrics(metric_dict)
                if with_tqdm:
                    tq.set_postfix(metric_dict)
            if not train:
                comet_experiment.log_metrics({
                    "overall_" + k : torch.mean(torch.as_tensor(batch_means[k])).item() for k in batch_means.keys()
                })
            