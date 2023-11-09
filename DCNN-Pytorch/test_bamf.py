import argparse
import shutil
import comet_ml
import deepracing_models.math_utils.bezier, deepracing_models.math_utils
from deepracing_models.nn_models.trajectory_prediction.lstm_based import BAMF
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

def assetkey(asset : dict):
    return asset["step"]
def test(**kwargs):
    experiment : str = kwargs["experiment"]
    tempdir : str = kwargs["tempdir"]
    workers : int = kwargs["workers"]
    batch_size : int = kwargs.get("batch_size", 1)
    gpu_index : int = kwargs.get("gpu", 0)
    
    api : comet_ml.API = comet_ml.API(api_key=os.getenv("COMET_API_KEY"))
    api_experiment : comet_ml.APIExperiment = api.get(workspace="electric-turtle", project_name="bamf", experiment=experiment)
    asset_list = api_experiment.get_asset_list()
    config_asset = None
    net_assets = []
    optimizer_assets = []
    for asset in asset_list:
        if asset["fileName"]=="config.yaml":
            config_asset = asset
        elif "optimizer_epoch_" in asset["fileName"]:
            optimizer_assets.append(asset)
        elif "model_" in asset["fileName"]:
            net_assets.append(asset)
    net_assets = sorted(net_assets, key=assetkey)
    optimizer_assets = sorted(optimizer_assets, key=assetkey)
    config_str = str(api_experiment.get_asset(config_asset["assetId"], return_type="binary"), encoding="ascii")
    config = yaml.safe_load(config_str)
    print(config)

    netconfig = config["network"]
    kbezier = netconfig["kbezier"]
    num_segments = netconfig["num_segments"]
    with_batchnorm = netconfig["with_batchnorm"]
    net : BAMF = BAMF( history_dimension = 6,
            num_segments = num_segments, 
            kbezier = kbezier,
            with_batchnorm = with_batchnorm)
    net_binary = api_experiment.get_asset(net_assets[-1]["assetId"], return_type="binary")
    net_bytesio = io.BytesIO(net_binary)
    net.load_state_dict(torch.load(net_bytesio, map_location="cpu"))
    net = net.eval().cuda(3)#.double()
    firstparam = next(net.parameters())
    dtype = firstparam.dtype
    device = firstparam.device
    data_config = config["data"]
    dsets : list[FD.TrajectoryPredictionDataset] = []
    for datadir in data_config["dirs"]: 
        dsets.extend(load_datasets_from_files(datadir, flag=SubsetFlag.VAL))
    dsetconfigs = [dset.metadata for dset in dsets]

    concat_set = torchdata.ConcatDataset(dsets)
    num_samples = len(concat_set)
    dataloader = torchdata.DataLoader(concat_set, num_workers=workers, batch_size=batch_size, pin_memory=True, shuffle=False)
    dataloader_enumerate = enumerate(dataloader)
    tq = tqdm.tqdm(dataloader_enumerate, desc="Yay", total=int(np.ceil(num_samples/batch_size)))
    ade_list = []
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
    curves_array : np.ndarray = np.zeros([num_samples, num_segments, kbezier+1, len(coordinate_idx_history)])
    history_array : np.ndarray = np.zeros([num_samples, Nhistory, len(coordinate_idx_history)])
    prediction_array : np.ndarray = np.zeros([num_samples, Nfuture, len(coordinate_idx_history)])
    ground_truth_array : np.ndarray = np.zeros_like(prediction_array)
    tsamp_ : torch.Tensor = torch.linspace(0.0, tfuture, steps=Nfuture, device=device, dtype=dtype)
    idxstart=0
    with torch.no_grad():
        for (i, dict_) in tq:
            datadict : dict[str,torch.Tensor] = dict_
            position_history = datadict["hist"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
            currentbatchdim = position_history.shape[0]
            idxend = idxstart+currentbatchdim
            history_array[idxstart:idxend,:] = position_history.cpu().numpy()

            vel_history = datadict["hist_vel"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
            quat_history = datadict["hist_quats"][:,:,quaternion_idx_history].to(device=device, dtype=dtype)
            quat_history = quat_history/torch.norm(quat_history, p=2.0, dim=-1, keepdim=True)
            quat_history = quat_history*(torch.sign(quat_history[:,:,-1])[...,None])

            position_future = datadict["fut"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
            ground_truth_array[idxstart:idxend,:] = position_future.cpu().numpy()
            tangents_future = datadict["fut_tangents"].to(device=device, dtype=dtype)#[:,:,coordinate_idx_history]
            quat_future = datadict["fut_quats"].to(device=device, dtype=dtype)
            rotmats_future = quaternionToMatrix(quat_future.view(-1,4)).view(currentbatchdim, -1, 3,3)
            upvecs_future = rotmats_future[...,-1]

            normals_future = torch.cross(upvecs_future, tangents_future)[:,:,coordinate_idx_history]
            normals_future /= torch.norm(normals_future, p=2.0, dim=-1, keepdim=True)

            tangents_future = tangents_future[:,:,coordinate_idx_history]
            tangents_future /= torch.norm(tangents_future, p=2.0, dim=-1, keepdim=True)

            # normal_future
            tangents_future = tangents_future/torch.norm(tangents_future, p=2.0, dim=-1, keepdim=True)
            vel_future = datadict["fut_vel"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)

            left_bound_positions = datadict["left_bd"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
            right_bound_positions = datadict["right_bd"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
            left_bound_tangents = datadict["left_bd_tangents"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
            right_bound_tangents = datadict["right_bd_tangents"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
            left_bound_tangents = left_bound_tangents/torch.norm(left_bound_tangents, p=2.0, dim=-1, keepdim=True)
            right_bound_tangents = right_bound_tangents/torch.norm(right_bound_tangents, p=2.0, dim=-1, keepdim=True)

            history_inputs = torch.cat([position_history, vel_history, quat_history], dim=-1)
            left_boundary_inputs = torch.cat([left_bound_positions, left_bound_tangents], dim=-1)
            right_boundary_inputs = torch.cat([right_bound_positions, right_bound_tangents], dim=-1)

            
            p0 = position_future[:,0]
            v0 = vel_future[:,0]        
        
            dt = dt_[None].expand(currentbatchdim, num_segments)
            velcurveout, poscurveout = net(history_inputs, left_boundary_inputs, right_boundary_inputs, dt, v0, p0=p0)
            curves_array[idxstart:idxend,:] = poscurveout.cpu().numpy()[:]
            tstart = tstart_[None].expand(currentbatchdim, num_segments)
            tsamp = tsamp_[None].expand(currentbatchdim, Nfuture)
            pout, _ = deepracing_models.math_utils.compositeBezierEval(tstart, dt, poscurveout, tsamp)
            prediction_array[idxstart:idxend,:] = pout.cpu().numpy()
            deltas = pout - position_future

            lateral_errors = torch.abs(torch.sum(deltas*tangents_future, dim=-1))
            longitudinal_errors = torch.abs(torch.sum(deltas*normals_future, dim=-1))

            lateral_error_list.extend(torch.mean(lateral_errors, dim=-1))
            longitudinal_error_list.extend(torch.mean(longitudinal_errors, dim=-1))
            
            delta_norms = torch.norm(deltas, p=2.0, dim=-1)
            mean_delta_norms = torch.mean(delta_norms, dim=-1)
            ade_list.extend(mean_delta_norms.cpu().numpy().tolist())
            tq.set_postfix({"current_error" : torch.mean(mean_delta_norms).item()})
            idxstart=idxend


    ade_array = torch.as_tensor(ade_list, dtype=torch.float64)
    print("ADE: %f" % (torch.mean(ade_array).item(),))
    lateral_error_array = torch.as_tensor(lateral_error_list, dtype=torch.float64)
    print("mean lateral error: %f" % (torch.mean(lateral_error_array).item(),))
    longitudinal_error_array = torch.as_tensor(longitudinal_error_list, dtype=torch.float64)
    print("mean longitudinal error: %f" % (torch.mean(longitudinal_error_array).item(),))

    results_dir = os.path.join(argdict["resultsdir"], experiment)
    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    with open(os.path.join(results_dir, "data.npz"), "wb") as f:
        np.savez(f, {
            "history" : history_array,
            "curves" : curves_array,
            "ground_truth" : prediction_array,
            "predictions" : ground_truth_array,
            "lateral_error" : lateral_error_array.cpu().numpy(),
            "longitudinal_error" : longitudinal_error_array.cpu().numpy(),
            "ade" : ade_array.cpu().numpy()
        })






if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Test Bezier version of MixNet")
    parser.add_argument("--experiment", type=str, required=True, help="Which comet experiment to load")
    parser.add_argument("--tempdir", type=str, default=os.path.join(os.environ["BIGTEMP"], "bamf"), help="Where temp space?!?!!?.")
    parser.add_argument("--resultsdir", type=str, default="/p/DeepRacing/bamf_results", help="Where put results?!?!??!")
    parser.add_argument("--workers", type=int, default=0, help="How many threads for data loading")
    parser.add_argument("--gpu", type=int, default=0, help="which gpu")
    args = parser.parse_args()
    argdict : dict = vars(args)
    argdict["batch_size"] = 512
    test(**argdict)