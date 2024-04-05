import argparse
import shutil
import time
import comet_ml
import deepracing_models.math_utils.bezier, deepracing_models.math_utils
from deepracing_models.nn_models.trajectory_prediction.lstm_based import BezierMixNet
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


def test(**kwargs):
    experiment : str = kwargs["experiment"]
    workers : int = kwargs["workers"]
    batch_size : int = kwargs.get("batch_size", 1)
    gpu_index : int = kwargs.get("gpu", 0)
    
    api : comet_ml.API = comet_ml.API(api_key=os.getenv("COMET_API_KEY"))
    api_experiment : comet_ml.APIExperiment = api.get(workspace="electric-turtle", project_name="mixnet-bezier", experiment=experiment)
    results_dir = os.path.join(argdict["resultsdir"], api_experiment.get_name())
    asset_list = api_experiment.get_asset_list()
    config_asset = None
    net_assets = []
    optimizer_assets = []
    for asset in asset_list:
        if asset["fileName"]=="config.yaml":
            config_asset = asset
        elif "optimizer_epoch_" in asset["fileName"]:
            optimizer_assets.append(asset)
        elif "network_epoch_" in asset["fileName"]:
            net_assets.append(asset)
    net_assets = sorted(net_assets, key=lambda a : a["step"])
    optimizer_assets = sorted(optimizer_assets, key=lambda a : a["step"])
    

    config_str = str(api_experiment.get_asset(config_asset["assetId"]), encoding="ascii")
    # print(config_str)
    config = yaml.safe_load(config_str)
    print(config)
    # if not os.path.isdir(experiment_dir):
    #     raise ValueError("PANIK!!!!!1!!!!ONEONE!!!!!")
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    net_file = os.path.join(results_dir, "model.pt")
    net_asset = api_experiment.get_asset(net_assets[-1]["assetId"], return_type="binary")
    with open(net_file, "wb") as f:
        f.write(net_asset)
    netconfig = config["net"]
    # netconfig[""]
    netconfig["gpu_index"]=gpu_index
    net : BezierMixNet = BezierMixNet(config["net"]).float().eval()
    firstparam = next(net.parameters())
    dtype = firstparam.dtype
    device = firstparam.device
    # with open(net_file, "rb") as f:
    state_dict = torch.load(net_file, map_location=device)
    net.load_state_dict(state_dict)
    # datadir = Path(data_config["dir"])
    # datadir="/p/DeepRacing/unpacked_datasets/local_fitting/v1/deepracing_standard"
    # datadir = datadir / "Monza_7_6_2023_16_23_11_trajectory_data" / "car_2"
    numsamples_prediction = None
    kbezier = netconfig["kbezier"]
    num_accel_sections : int = netconfig["acc_decoder"]["num_acc_sections"]
    dsetconfigs : list[dict] = []
    data_config = config["data"]
    dsets : list[FD.TrajectoryPredictionDataset] = []
    for datadir in data_config["dirs"]:
        dsets.extend(load_datasets_from_files(datadir, kbezier=kbezier, flag=SubsetFlag.TEST))
    dsetconfigs = [dset.metadata for dset in dsets]
    numsamples_history = dsetconfigs[0]["numsamples_history"]
    numsamples_prediction = dsetconfigs[0]["numsamples_prediction"]
    prediction_totaltime = dsetconfigs[0]["predictiontime"]

    tsamp = torch.linspace(0.0, prediction_totaltime, dtype=dtype, device=device, steps=numsamples_prediction)
    tswitchingpoints = torch.linspace(0.0, prediction_totaltime, dtype=dtype, device=device, steps=num_accel_sections+1)
    dt = tswitchingpoints[1:] - tswitchingpoints[:-1]
    kbeziervel = netconfig["acc_decoder"]["kbeziervel"]


    concat_set = torchdata.ConcatDataset(dsets)
    num_samples = len(concat_set)
    dataloader = torchdata.DataLoader(concat_set, num_workers=workers, batch_size=batch_size, pin_memory=True, shuffle=False)
    dataloader_enumerate = enumerate(dataloader)
    tq = tqdm.tqdm(dataloader_enumerate, desc="Yay", total=int(np.ceil(num_samples/batch_size)))
    ade_list = []
    fde_list = []
    lateral_error_list = []
    longitudinal_error_list = []
    computation_time_list = []
    history_array : np.ndarray = np.zeros([num_samples, numsamples_history, 3])
    prediction_array : np.ndarray = np.zeros([num_samples, numsamples_prediction, 3])
    ground_truth_array : np.ndarray = np.zeros_like(prediction_array)
    coordinate_idx_history = [0,1]
    input_embedding = netconfig["input_embedding"]
    boundary_embedding = netconfig["boundary_embedding"]
    idx = 0
    for (i, dict_) in tq:
        datadict : dict[str,torch.Tensor] = dict_

        position_history = datadict["hist"]
        vel_history = datadict["hist_vel"]

        position_future = datadict["fut"]
        tangent_future = datadict["fut_tangents"]
        speed_future = datadict["fut_speed"]
        future_arclength = datadict["future_arclength"]

        left_bound_input = datadict["left_bd"]
        right_bound_input = datadict["right_bd"]

        bcurves_r = datadict["reference_curves"]

        vel_history = vel_history.cuda(gpu_index).type(dtype)
        position_history = position_history.cuda(gpu_index).type(dtype)
        position_future = position_future.cuda(gpu_index).type(dtype)
        speed_future = speed_future.cuda(gpu_index).type(dtype)
        future_arclength = future_arclength.cuda(gpu_index).type(dtype)
        bcurves_r = bcurves_r.cuda(gpu_index).type(dtype)
        tangent_future = tangent_future.cuda(gpu_index).type(dtype)

        currentbatchsize = int(position_history.shape[0])
        with torch.no_grad():
            tick = time.time()
            state_input = position_history[:,:,[0,1]]
            if input_embedding["velocity"]:
                state_input = torch.cat([state_input, vel_history[:,:,[0,1]]], dim=-1)
            quat_history = datadict["hist_quats"].cuda(gpu_index).type(dtype)
            if input_embedding["quaternion"]:
                if state_input.shape[-1]==3:
                    quat_select = quat_history
                else:
                    quat_select = quat_history[:,:,[2,3]]
                    quat_select = quat_select/torch.norm(quat_select, p=2.0, dim=-1, keepdim=True)
                realparts = quat_select[:,:,-1]
                quat_select[realparts<0.0]*=-1.0
                state_input = torch.cat([state_input, quat_select], dim=-1)

            if input_embedding["heading_angle"]:
                quat_history_yaw_only = quat_history*torch.as_tensor([[[0.0, 0.0, 1.0, 1.0],],], device=device, dtype=dtype)#[None,None]
                quat_history_yaw_only/=torch.norm(quat_history_yaw_only, p=2.0, dim=-1, keepdim=True)
                qz = quat_history_yaw_only[:,:,2]
                qw = quat_history_yaw_only[:,:,3]
                headings = 2.0*torch.atan2(qz,qw).unsqueeze(-1)
                state_input = torch.cat([state_input, headings], dim=-1)

            left_bound_input = datadict["left_bd"][:,:,coordinate_idx_history].cuda(gpu_index).type(dtype)
            right_bound_input = datadict["right_bd"][:,:,coordinate_idx_history].cuda(gpu_index).type(dtype)
            if boundary_embedding["tangent"]:

                left_bound_tangents = datadict["left_bd_tangents"][:,:,coordinate_idx_history].cuda(gpu_index).type(dtype)
                left_bound_tangents = left_bound_tangents/torch.norm(left_bound_tangents, p=2.0, dim=-1, keepdim=True)

                right_bound_tangents = datadict["right_bd_tangents"][:,:,coordinate_idx_history].cuda(gpu_index).type(dtype)
                right_bound_tangents = right_bound_tangents/torch.norm(right_bound_tangents, p=2.0, dim=-1, keepdim=True)

                left_bound_input = torch.cat([left_bound_input, left_bound_tangents], dim=-1)
                right_bound_input = torch.cat([right_bound_input, right_bound_tangents], dim=-1)

            # print(tangent_future[:,0])
            (mix_out_, acc_out_) = net(state_input, left_bound_input, right_bound_input)
            one = torch.ones_like(speed_future[0,0])
            mix_out = torch.clamp(mix_out_, -0.5*one, 1.5*one)
            # + speed_future[:,0].unsqueeze(-1)
            acc_out = acc_out_ + speed_future[:,0].unsqueeze(-1)
            # acc_out = torch.clamp(acc_out_ + speed_future[:,0].unsqueeze(-1), 5.0*one, 110.0*one)
            

            coefs_inferred = torch.zeros(currentbatchsize, num_accel_sections, kbeziervel+1, dtype=acc_out.dtype, device=acc_out.device)
            coefs_inferred[:,0,0] = speed_future[:,0]
            if kbeziervel == 3:
                setter_idx = [True, True, True, False]
                coefs_inferred[:,0,[1,2]] = acc_out[:,[0,1]]
                coefs_inferred[:,1:,setter_idx] = acc_out[:,2:-1].view(coefs_inferred[:,1:,setter_idx].shape)
                coefs_inferred[:,:-1,-1] = coefs_inferred[:,1:,0]   
                coefs_inferred[:,-1,-1] = acc_out[:,-1] 
            elif kbeziervel == 2:
                setter_idx = [True, True, False]
                coefs_inferred[:,0,1] = acc_out[:,0]
                coefs_inferred[:,1:,setter_idx] = acc_out[:,1:-1].view(coefs_inferred[:,1:,setter_idx].shape)
                coefs_inferred[:,:-1,-1] = coefs_inferred[:,1:,0]   
                coefs_inferred[:,-1,-1] = acc_out[:,-1] 
            elif kbeziervel == 1:
                setter_idx = [True, False]
                coefs_inferred[:,1:,setter_idx] = acc_out[:,0:-1].view(coefs_inferred[:,1:,setter_idx].shape)
                coefs_inferred[:,:-1,-1] = coefs_inferred[:,1:,0]   
                coefs_inferred[:,-1,-1] = acc_out[:,-1] 
            else:
                raise ValueError("Only order 1, 2, or 3 velocity segments are supported")
            

            tstart_batch = tswitchingpoints[:-1].unsqueeze(0).expand(currentbatchsize, num_accel_sections)
            dt_batch = dt.unsqueeze(0).expand(currentbatchsize, num_accel_sections)
            teval_batch = tsamp.unsqueeze(0).expand(currentbatchsize, numsamples_prediction)
            speed_profile_out, idxbuckets = deepracing_models.math_utils.compositeBezierEval(tstart_batch, dt_batch, coefs_inferred.unsqueeze(-1), teval_batch)
            


            coefs_antiderivative = deepracing_models.math_utils.compositeBezierAntiderivative(coefs_inferred.unsqueeze(-1), dt_batch)

            mixed_control_points = torch.sum(bcurves_r[:,:,:,[0,1]]*mix_out[:,:,None,None], dim=1)
            mcp_deltar : torch.Tensor = deepracing_models.math_utils.bezierArcLength(mixed_control_points, num_segments = 10)

            # delta_r : torch.Tensor = arclengths_pred[:,-1] - arclengths_pred[:,0]
            # delta_r : torch.Tensor = future_arclength[:,-1] - future_arclength[:,0]
            # delta_r : torch.Tensor = deepracing_models.math_utils.bezierArcLength(mixed_control_points, quadrature_order=9)

            known_control_points : torch.Tensor = torch.zeros_like(bcurves_r[:,0,:2,[0,1]])
            known_control_points[:,0] = position_future[:,0,[0,1]]
            known_control_points[:,1] = (mcp_deltar[:,None]/kbezier)*tangent_future[:,0,[0,1]]
            # known_control_points[:,1] = known_control_points[:,0] + (delta_r[:,None]/kbezier)*tangent_future[:,0,[0,1]]

            predicted_bcurve = torch.cat([known_control_points, mixed_control_points[:,2:]], dim=1) 
            pred_deltar : torch.Tensor = deepracing_models.math_utils.bezierArcLength(predicted_bcurve, num_segments = 10)

            arclengths_pred, _ = deepracing_models.math_utils.compositeBezierEval(tstart_batch, dt_batch, coefs_antiderivative, teval_batch, idxbuckets=idxbuckets)

            arclengths_deltar = arclengths_pred[:,-1]
            idx_clip = arclengths_deltar>pred_deltar
            arclengths_pred[idx_clip]*=(pred_deltar[idx_clip]/arclengths_deltar[idx_clip])[:,None]

            arclengths_pred_s = arclengths_pred/arclengths_pred[:,-1,None]
            Marclengths_pred : torch.Tensor = deepracing_models.math_utils.bezierM(arclengths_pred_s, kbezier)



            pointsout : torch.Tensor = torch.matmul(Marclengths_pred, predicted_bcurve)
            tock = time.time()
            computation_time_list.append(float(tock - tick))
            tq.set_postfix({"computation_time" : computation_time_list[-1]})
            displacements : torch.Tensor = pointsout[:,:,[0,1]] - position_future[:,:,[0,1]]
            displacement_norms = torch.norm(displacements, p=2.0, dim=-1)

            tangent_future_xy = tangent_future[:,:,[0,1]]/torch.norm(tangent_future[:,:,[0,1]], p=2.0, dim=-1, keepdim=True)
            normal_future_xy = tangent_future_xy[:,:,[1,0]].clone()
            normal_future_xy[:,:,0]*=-1.0

            rotmats_decomposition = torch.stack([tangent_future_xy, normal_future_xy], axis=3).transpose(-2,-1)
            translations_decomposition = torch.matmul(rotmats_decomposition, -position_future[:,:,[0,1]].unsqueeze(-1)).squeeze(-1)

            decompositions = torch.matmul(rotmats_decomposition, pointsout[:,:,[0,1]].unsqueeze(-1)).squeeze(-1) + translations_decomposition
            longitudinal_errors = torch.abs(decompositions[:,:,0])
            lateral_errors = torch.abs(decompositions[:,:,1])

            lateral_error_list+=torch.mean(lateral_errors, dim=1).cpu().numpy().tolist()
            longitudinal_error_list+=torch.mean(longitudinal_errors, dim=1).cpu().numpy().tolist()
            current_batch_size = position_future.shape[0]

            history_array[idx:idx+current_batch_size, :] = position_history.cpu().numpy()[:]
            ground_truth_array[idx:idx+current_batch_size, :] = position_future.cpu().numpy()[:]

            dim_prediction = pointsout.shape[-1]
            prediction_array[idx:idx+current_batch_size, :, :dim_prediction] = pointsout.cpu().numpy()[:]
            
            idx+=current_batch_size

            ade : torch.Tensor = torch.mean(displacement_norms, dim=-1)
            ade_list+=ade.cpu().numpy().tolist()
            fde : torch.Tensor = torch.norm(displacements[:,-1], dim=1)
            fde_list+=fde.cpu().numpy().tolist()
        
    lateral_error_array = torch.as_tensor(lateral_error_list, dtype=torch.float64)
    longitudinal_error_array = torch.as_tensor(longitudinal_error_list, dtype=torch.float64)
    ade_array = torch.as_tensor(ade_list, dtype=torch.float64)
    fde_array = torch.as_tensor(fde_list, dtype=ade_array.dtype)
    computation_time_array = torch.as_tensor(computation_time_list, dtype=torch.float64)



    resultsdict = {
        "lateral_error" : lateral_error_array,
        "longitudinal_error" : longitudinal_error_array,
        "ade" : ade_array,
        "fde" : fde_array,
        "history" : history_array,
        "ground_truth" : ground_truth_array,
        "predictions" : prediction_array,
        # "future_left_bd" : future_left_bd_array,
        # "future_right_bd" : future_right_bd_array,
        "computation_time" : computation_time_array
    }
    # summary_dict = {
    #     "ade": torch.mean(ade_array).item(),
    #     "longitudinal_error": torch.mean(longitudinal_error_array).item(),
    #     "lateral_error": torch.mean(lateral_error_array).item(),
    #     "computation_time": torch.mean(computation_time_array).item()
    # }
    error_keys = {"ade", "fde", "lateral_error", "longitudinal_error", "computation_time"}
    summary_dict = {
        k : {
            "mean" : torch.mean(v).item(),
            "min" : torch.min(v).item(),
            "max" : torch.max(v).item(),
            "stdev" : torch.std(v).item()
        } 
        for (k,v) in resultsdict.items() if k in error_keys
    }
    print(summary_dict)
    # exit(0)
    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    with open(os.path.join(results_dir, "summary.yaml"), "w") as f:
        yaml.safe_dump(summary_dict, f)
    with open(os.path.join(results_dir, "data.npz"), "wb") as f:
        np.savez(f, **{
            "history" : history_array,
            "ground_truth" : ground_truth_array,
            "predictions" : prediction_array,
            "lateral_error" : lateral_error_array.cpu().numpy(),
            "longitudinal_error" : longitudinal_error_array.cpu().numpy(),
            "ade" : ade_array.cpu().numpy()
        })






if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Test Bezier version of MixNet")
    parser.add_argument("--experiment", type=str, required=True, help="Which comet experiment to load")
    parser.add_argument("--tempdir", type=str, default="/bigtemp/ttw2xk/mixnet_bezier_dump", help="Where temp space?!?!!?.")
    parser.add_argument("--resultsdir", type=str, default="/p/DeepRacing/mixnet_bezier_results", help="Where put results?!?!??!")
    parser.add_argument("--workers", type=int, default=0, help="How many threads for data loading")
    parser.add_argument("--gpu", type=int, default=0, help="which gpu")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    args = parser.parse_args()
    argdict : dict = vars(args)
    test(**argdict)