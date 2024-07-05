import argparse
import shutil
import time
import comet_ml
import deepracing_models.math_utils.bezier, deepracing_models.math_utils
from deepracing_models.nn_models.trajectory_prediction.lstm_based import BARTE
from deepracing_models.data_loading import file_datasets as FD, SubsetFlag 
from deepracing_models.data_loading.utils.file_utils import load_datasets_from_files 
import torch.utils.data as torchdata
import yaml
import os
import io
import numpy as np
import pickle as pkl
import tqdm
import matplotlib.figure
import matplotlib.pyplot as plt
import glob
import multiprocessing, multiprocessing.pool
import traceback
import sys
from datetime import datetime
import torch
from pathlib import Path
from deepracing_models.math_utils.rotations import quaternionToMatrix
import matplotlib.axes, matplotlib.figure


def test(**kwargs):
    experiment : str = kwargs["experiment"]
    workers : int = kwargs["workers"]
    batch_size : int = kwargs.get("batch_size", 1)
    gpu_index : int = kwargs.get("gpu", 0)
    
    api : comet_ml.API = comet_ml.API(api_key=os.getenv("COMET_API_KEY"))
    api_experiment : comet_ml.APIExperiment = api.get(workspace="electric-turtle", project_name="bamf", experiment=experiment)
    asset_list = api_experiment.get_asset_list()
    config_asset = None
    net_assets = []
    for asset in asset_list:
        if asset["fileName"]=="config.yaml":
            config_asset = asset
        elif "model_" in asset["fileName"]:
            net_assets.append(asset)
    net_assets.sort(key = lambda x : x["step"])
    # net_assets = sorted(net_assets, key=lambda a : a["step"])
    print("Downloading config", flush=True)
    config_str = str(api_experiment.get_asset(config_asset["assetId"], return_type="binary"), encoding="ascii")
    config = yaml.safe_load(config_str)
    netconfig = config["network"]
    print(netconfig)
    

    kbezier = netconfig["kbezier"]
    num_segments = netconfig["num_segments"]
    with_batchnorm = netconfig["with_batchnorm"]
    heading_encoding = netconfig.get("heading_encoding", "quaternion")
    if heading_encoding=="angle":
        print("Using heading angle as orientation input")
        history_dimension = 5
    elif heading_encoding=="quaternion":
        print("Using quaternion as orientation input")
        history_dimension = 6
    else:
        raise ValueError("Unknown heading encoding: %s" % (heading_encoding,))
    net : BARTE = BARTE( history_dimension = history_dimension,
            num_segments = num_segments, 
            kbezier = kbezier,
            with_batchnorm = with_batchnorm)
    trainerconfig : dict = config["trainer"]
    if trainerconfig["float32"]:
        net = net.float()
    else:
        net = net.double()
    # net_asset = ([a for a in net_assets if a["fileName"] == "model_148.pt"])[0]
    net_asset = net_assets[-1]
    print("Downloading model file: %s" % (net_asset["fileName"],))
    net_binary = api_experiment.get_asset(net_asset["assetId"], return_type="binary")
    results_dir = os.path.join(argdict["resultsdir"], api_experiment.get_name())
    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(os.path.join(results_dir, "plots"))
    net_bytesio : io.BytesIO = io.BytesIO(net_binary)
    net.load_state_dict(torch.load(net_bytesio))
    net = net.to(device = torch.device("cuda:%d" % (gpu_index,))).eval()
    firstparam = next(net.parameters())
    dtype = firstparam.dtype
    device = firstparam.device
    data_config = config["data"]
    print("Loading data", flush=True)
    keys : set = {
        "thistory",
        "hist",
        "hist_quats",
        "hist_vel",
        "tfuture",
        "fut",
        "fut_quats",
        "fut_vel",
        "fut_tangents",
        "left_bd",
        "left_bd_tangents",
        "right_bd",
        "right_bd_tangents",
    }
    
    dsets : list[FD.TrajectoryPredictionDataset] = []
    for datadir in data_config["dirs"]: 
        dsets += load_datasets_from_files(datadir, keys=keys, flag=SubsetFlag.TEST, dtype=np.float64) #

    concat_set = torchdata.ConcatDataset(dsets)
    num_samples = len(concat_set)
    dataloader = torchdata.DataLoader(concat_set, num_workers=workers, batch_size=batch_size, pin_memory=True, shuffle=False)
    dataloader_enumerate = enumerate(dataloader)
    tq = tqdm.tqdm(dataloader_enumerate, desc="Yay", total=int(np.ceil(num_samples/batch_size)))
    ade_list = []
    fde_list = []
    lateral_error_list = []
    longitudinal_error_list = []
    coordinate_idx_history = [0,1]
    quaternion_idx_history = [2,3]
    Nhistory = dsets[0].metadata["numsamples_history"]
    Nfuture = dsets[0].metadata["numsamples_prediction"]
    tfuture = dsets[0].metadata["predictiontime"]
    tsegs : torch.Tensor = torch.linspace(0.0, tfuture, steps=num_segments+1, device=device, dtype=dtype)
    tstart_ = tsegs[:-1]
    dt_ = tsegs[1:] - tstart_
    tsamp_ : torch.Tensor = torch.linspace(0.0, tfuture, steps=Nfuture, device=device, dtype=dtype)
    curves_array : np.ndarray = np.zeros([num_samples, num_segments, kbezier+1, len(coordinate_idx_history)])
    history_array : np.ndarray = np.zeros([num_samples, Nhistory, len(coordinate_idx_history)])
    hist_vel_array : np.ndarray = np.zeros([num_samples, Nhistory, len(coordinate_idx_history)])
    future_vel_array : np.ndarray = np.zeros([num_samples, Nhistory, len(coordinate_idx_history)])
    prediction_array : np.ndarray = np.zeros([num_samples, Nfuture, len(coordinate_idx_history)])
    ground_truth_array : np.ndarray = np.zeros_like(prediction_array)
    computation_time_list : list[float] = []
    idxstart=0
    with torch.no_grad():
        for (i, dict_) in tq:
            datadict : dict[str,torch.Tensor] = dict_
            position_history = datadict["hist"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
            vel_history = datadict["hist_vel"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
            quat_history = datadict["hist_quats"][:,:,quaternion_idx_history].to(device=device, dtype=dtype)
            quat_history = quat_history*(torch.sign(quat_history[:,:,-1])[...,None])
            quat_history = quat_history/torch.norm(quat_history, p=2.0, dim=-1, keepdim=True)
            if heading_encoding=="quaternion":
                quat_input = quat_history
            elif heading_encoding=="angle":
                qz = quat_history[:,:,-2]
                qw = quat_history[:,:,-1]
                quat_input = 2.0*torch.atan2(qz,qw).unsqueeze(-1)

            position_future = datadict["fut"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
            vel_future = datadict["fut_vel"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
            tangents_future = datadict["fut_tangents"].to(device=device, dtype=dtype)#[:,:,coordinate_idx_history]
            quat_future = datadict["fut_quats"].to(device=device, dtype=dtype)
            quat_future = quat_future*(torch.sign(quat_future[:,:,-1])[...,None])
            quat_future = quat_future/torch.norm(quat_future, p=2.0, dim=-1, keepdim=True)
            currentbatchdim = position_history.shape[0]
            rotmats_future = quaternionToMatrix(quat_future.view(-1,4)).view(currentbatchdim, -1, 3,3)
            upvecs_future = rotmats_future[...,-1]

            normals_future = torch.cross(upvecs_future, tangents_future)[:,:,coordinate_idx_history]
            normals_future /= torch.norm(normals_future, p=2.0, dim=-1, keepdim=True)

            tangents_future = tangents_future[:,:,coordinate_idx_history]
            tangents_future /= torch.norm(tangents_future, p=2.0, dim=-1, keepdim=True)

            tangents_future = tangents_future/torch.norm(tangents_future, p=2.0, dim=-1, keepdim=True)

            left_bound_positions = datadict["left_bd"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
            right_bound_positions = datadict["right_bd"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
            left_bound_tangents = datadict["left_bd_tangents"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
            right_bound_tangents = datadict["right_bd_tangents"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
            left_bound_tangents = left_bound_tangents/torch.norm(left_bound_tangents, p=2.0, dim=-1, keepdim=True)
            right_bound_tangents = right_bound_tangents/torch.norm(right_bound_tangents, p=2.0, dim=-1, keepdim=True)

            history_inputs = torch.cat([position_history, vel_history, quat_input], dim=-1)
            left_boundary_inputs = torch.cat([left_bound_positions, left_bound_tangents], dim=-1)
            right_boundary_inputs = torch.cat([right_bound_positions, right_bound_tangents], dim=-1)

            p0 = position_future[:,0]
            v0 = vel_future[:,0]        
            currentbatchdim = position_history.shape[0]
            dt = dt_[None].expand(currentbatchdim, num_segments)
            tstart = tstart_[None].expand(currentbatchdim, num_segments)
            tsamp = tsamp_[None].expand(currentbatchdim, Nfuture)

            tick = time.time()
            velcurveout, poscurveout = net(history_inputs, left_boundary_inputs, right_boundary_inputs, dt, v0, p0=p0)
            pout, _ = deepracing_models.math_utils.compositeBezierEval(tstart, dt, poscurveout, tsamp)
            tock = time.time()
            computation_time_list.append(tock - tick)
            deltas = pout - position_future
            de = torch.norm(deltas, p=2.0, dim=2)
            # de = torch.sum(torch.square(deltas), dim=2)
            fde = de[:,-1].clone()
            minde, minde_idx = torch.min(de, dim=1)
            ade = torch.mean(de, dim=1)
            maxde, maxde_idx = torch.max(de, dim=1)

            lateral_errors = torch.abs(torch.sum(deltas*normals_future, dim=-1))
            longitudinal_errors = torch.abs(torch.sum(deltas*tangents_future, dim=-1))

            lateral_error_list.extend(torch.mean(lateral_errors, dim=-1))
            longitudinal_error_list.extend(torch.mean(longitudinal_errors, dim=-1))

            tq.set_postfix({
                "ade" : torch.mean(ade).item(),
                "minde" : torch.mean(minde).item(),
                "maxde" : torch.mean(maxde).item()
                })
            ade_list.extend(ade.cpu().numpy().tolist())
            fde_list.extend(fde.cpu().numpy().tolist())
            idxend = idxstart+currentbatchdim
            curves_array[idxstart:idxend,:] = poscurveout.cpu().numpy()[:]
            prediction_array[idxstart:idxend,:] = pout.cpu().numpy()[:]
            ground_truth_array[idxstart:idxend,:] = position_future.cpu().numpy()[:]
            future_vel_array[idxstart:idxend,:] = vel_future.cpu().numpy()[:]
            history_array[idxstart:idxend,:] = position_history.cpu().numpy()[:]
            hist_vel_array[idxstart:idxend,:] = vel_history.cpu().numpy()[:]
            idxstart=idxend


    results_dict : dict = dict()
    results_dict.update(config)
    results_dict = dict()
    ade_array = torch.as_tensor(ade_list, dtype=dtype)
    results_dict["ade"] = {
        "mean" : torch.mean(ade_array).item(), 
        "stdev" : torch.std(ade_array).item(), 
        "max" : torch.max(ade_array).item()
    }
    fde_array = torch.as_tensor(fde_list, dtype=dtype)
    results_dict["fde"] = {
        "mean" : torch.mean(fde_array).item(), 
        "stdev" : torch.std(fde_array).item(), 
        "max" : torch.max(fde_array).item()
    }
    lateral_error_array = torch.as_tensor(lateral_error_list, dtype=dtype)
    results_dict["lateral_error"] = {
        "mean" : torch.mean(lateral_error_array).item(), 
        "stdev" : torch.std(lateral_error_array).item(), 
        "max" : torch.max(lateral_error_array).item()
    }
    longitudinal_error_array = torch.as_tensor(longitudinal_error_list, dtype=dtype)
    results_dict["longitudinal_error"] = {
        "mean" : torch.mean(longitudinal_error_array).item(), 
        "stdev" : torch.std(longitudinal_error_array).item(), 
        "max" : torch.max(longitudinal_error_array).item()
    }

    print(results_dict)


    with open(os.path.join(results_dir, "summary.yaml"), "w") as f:
        yaml.dump(results_dict, f, Dumper=yaml.SafeDumper)
    with open(os.path.join(results_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.SafeDumper)

    with open(os.path.join(results_dir, "data.npz"), "wb") as f:
        np.savez(f, **{
            "history" : history_array,
            "history_vel" : hist_vel_array,
            "curves" : curves_array,
            "ground_truth" : ground_truth_array,
            "ground_truth_vel" : future_vel_array,
            "predictions" : prediction_array,
            "lateral_error" : lateral_error_array.cpu().numpy(),
            "longitudinal_error" : longitudinal_error_array.cpu().numpy(),
            "ade" : ade_array.cpu().numpy()
        })






if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Test Bezier version of MixNet")
    parser.add_argument("--experiment", type=str, required=True, help="Which comet experiment to load")
    parser.add_argument("--resultsdir", type=str, default="/p/DeepRacing/barte_results", help="Where put results?!?!??!")
    parser.add_argument("--workers", type=int, default=0, help="How many threads for data loading")
    parser.add_argument("--gpu", type=int, default=0, help="which gpu")
    args = parser.parse_args()
    argdict : dict = vars(args)
    argdict["batch_size"] = 16
    test(**argdict)