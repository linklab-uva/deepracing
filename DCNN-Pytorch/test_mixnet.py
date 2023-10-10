import argparse
import shutil
import comet_ml
import deepracing_models.math_utils.bezier, deepracing_models.math_utils
from mix_net.src.mix_net import MixNet
from deepracing_models.data_loading import file_datasets as FD, SubsetFlag
from deepracing_models.data_loading.utils.file_utils import load_datasets_from_files 
import torch.utils.data as torchdata
import yaml, json
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
import deepracing
from scipy.spatial.transform import Rotation

def assetkey(asset : dict):
    return asset["step"]
def test(**kwargs):
    experiment : str = kwargs["experiment"]
    tempdir : str = kwargs["tempdir"]
    workers : int = kwargs["workers"]
    batch_size : int = kwargs.get("batch_size", 1)
    gpu_index : int = kwargs.get("gpu_index", 0)
    
    experiment_dir = os.path.join(tempdir, experiment)
    api : comet_ml.API = comet_ml.API(api_key=os.getenv("COMET_API_KEY"))
    api_experiment : comet_ml.APIExperiment = api.get(workspace="electric-turtle", project_name="mixnet-deepracing", experiment=experiment)
    asset_list = api_experiment.get_asset_list()
    trainer_config_asset = None
    net_config_asset = None
    net_assets = []
    optimizer_assets = []
    for asset in asset_list:
        if asset["fileName"]=="trainer_params.json":
            trainer_config_asset = asset
        elif asset["fileName"]=="net_params.json":
            net_config_asset = asset
        elif "optimizer_epoch_" in asset["fileName"]:
            optimizer_assets.append(asset)
        elif "model_epoch_" in asset["fileName"]:
            net_assets.append(asset)
    net_assets = sorted(net_assets, key=assetkey)
    optimizer_assets = sorted(optimizer_assets, key=assetkey)
    
    trainer_config_str = str(api_experiment.get_asset(trainer_config_asset["assetId"]), encoding="ascii")
    trainerconfig = json.loads(trainer_config_str)
    trainerconfig["data"]["path"] = \
        "/p/DeepRacing/unpacked_datasets/v1/online_multiplayer/deepracing_standard"

    net_config_str = str(api_experiment.get_asset(net_config_asset["assetId"]), encoding="ascii")
    netconfig = json.loads(net_config_str)
    print(netconfig)
    print(netconfig.keys())
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)
    net_file = os.path.join(experiment_dir, "net.pt")
    if not os.path.isfile(net_file):
        net_asset = api_experiment.get_asset(net_assets[-1]["assetId"], return_type="binary")
        with open(net_file, "wb") as f:
            f.write(net_asset)
    netconfig["use_cuda"]=gpu_index
    device = torch.device("cuda:%d" % (gpu_index,))
    net : MixNet = MixNet(netconfig).double().eval()
    with open(net_file, "rb") as f:
        state_dict = torch.load(f, map_location=device)
    net.load_state_dict(state_dict)
    firstparam = next(net.parameters())
    dtype = firstparam.dtype
    datadir = trainerconfig["data"]["path"]
    numsamples_prediction = None
    num_accel_sections : int = netconfig["acc_decoder"]["num_acc_sections"]
    dsetconfigs : list[dict] = []

    # position_history = datadict["hist"]#[:,:,[0,1]]
    # position_future = datadict["fut"]#[:,:,[0,1]]
    # tangents_future = datadict["fut_tangents"]#[:,:,[0,1]]

    # current_positions_full = datadict["current_position"]
    # current_orientations_full = datadict["current_orientation"]

    # rotations = Rotation.from_quat(current_orientations_full.detach().cpu().numpy())

    # left_bound_input = datadict["left_bd"][:,:,[0,1]]
    # right_bound_input = datadict["right_bd"][:,:,[0,1]]

    # tracknames = datadict["trackname"]

    # left_bound_label = datadict["future_left_bd"][:,:,[0,1]]
    # right_bound_label = datadict["future_right_bd"][:,:,[0,1]]
    # center_line_label = datadict["future_centerline"][:,:,[0,1]]
    # optimal_line_label = datadict["future_raceline"][:,:,[0,1]]
    keys = {"hist", "fut", "fut_tangents", "current_position", "current_orientation", 
            "left_bd", "right_bd", "future_left_bd", "future_right_bd", "future_centerline", "future_raceline"}
    dsets : list[FD.TrajectoryPredictionDataset] = load_datasets_from_files(datadir, keys=keys)
    track_dict : dict = dict()
    searchdirs = []
    try:
        searchdirs+=os.environ["F1_MAP_DIRS"].split(os.pathsep)
    except KeyError:
        pass
    try:
        import ament_index_python
        deepracing_launch_dir = ament_index_python.get_package_share_directory("deepracing_launch")
        searchdirs.append(os.path.join(deepracing_launch_dir, "maps"))
    except ImportError:
        pass
    except ament_index_python.PackageNotFoundError:
        pass

    for dset in dsets:
        current_track = dset.metadata["trackname"]
        dsetconfigs.append(dset.metadata.copy())
        if not (current_track in track_dict.keys()):
            with_z = False
            trackmap = deepracing.searchForTrackmap(current_track, searchdirs)
            lb_helper = trackmap.getPathHelper("outer_boundary_optimized_safe", with_z = with_z)
            rb_helper = trackmap.getPathHelper("inner_boundary_optimized_safe", with_z = with_z)
            cl_helper = trackmap.getPathHelper("centerline_optimized_safe", with_z = with_z)
            ol_helper = trackmap.getPathHelper("raceline_optimized_safe", with_z = with_z)

            drdesired = 0.02
            Nsamp = int(np.mean([ lb_helper.distances[-1]/drdesired,
                              rb_helper.distances[-1]/drdesired,
                              cl_helper.distances[-1]/drdesired,
                              ol_helper.distances[-1]/drdesired,
                            ]).round())
            
            lb_np = lb_helper.spline(np.linspace(0.0, lb_helper.distances[-1]-drdesired, num=Nsamp))
            
            rb_np = rb_helper.spline(np.linspace(0.0, rb_helper.distances[-1]-drdesired, num=Nsamp))

            cl_np = cl_helper.spline(np.linspace(0.0, cl_helper.distances[-1]-drdesired, num=Nsamp))
            
            ol_np = ol_helper.spline(np.linspace(0.0, ol_helper.distances[-1]-drdesired, num=Nsamp))

            track_dict[current_track] = torch.as_tensor( 
                np.stack([
                        lb_np,
                        rb_np,
                        cl_np,
                        ol_np,
                    ], axis=0
                    ), 
            dtype=firstparam.dtype, device=device)
    numsamples_history = dsetconfigs[0]["numsamples_history"]
    numsamples_prediction = dsetconfigs[0]["numsamples_prediction"]
    prediction_totaltime = dsetconfigs[0]["predictiontime"]

    tsamp = torch.linspace(0.0, prediction_totaltime, dtype=dtype, device=device, steps=numsamples_prediction)
    ssamp = tsamp/tsamp[-1]
    time_spacing = (tsamp[1]-tsamp[0]).item()
    tswitchingpoints = torch.linspace(0.0, prediction_totaltime, dtype=dtype, device=device, steps=num_accel_sections+1)
    dt = tswitchingpoints[1:] - tswitchingpoints[:-1]
    kbeziervel = 3
    N = numsamples_prediction - 1
    M = netconfig["acc_decoder"]["num_acc_sections"]
    _time_profile_matrix = torch.zeros(
        (N, M), dtype=firstparam.dtype, device=device
    )
    ratio = int(N/M) #10 for standard configuration
    for i in range(M):
        _time_profile_matrix[(i * ratio) : ((i + 1) * ratio), i] = torch.linspace(
            time_spacing, 1.0, ratio, dtype=firstparam.dtype, device=device
        )

        _time_profile_matrix[((i + 1) * ratio) :, i] = 1.0

    print(_time_profile_matrix)

    concat_set = torchdata.ConcatDataset(dsets)
    num_samples = len(concat_set)
    dataloader = torchdata.DataLoader(concat_set, num_workers=workers, batch_size=batch_size, pin_memory=True, shuffle=False)
    dataloader_enumerate = enumerate(dataloader)
    tq = tqdm.tqdm(dataloader_enumerate, desc="Yay", total=int(np.ceil(num_samples/batch_size)))
    ade_list = []
    lateral_error_list = []
    longitudinal_error_list = []
    
    history_array : np.ndarray = np.zeros([num_samples, numsamples_history, 3], dtype=np.float64)
    prediction_array : np.ndarray = np.zeros([num_samples, numsamples_prediction, 3], dtype=np.float64)
    future_left_bd_array : np.ndarray = np.zeros([num_samples, numsamples_prediction, 3], dtype=np.float64)
    future_right_bd_array : np.ndarray = np.zeros([num_samples, numsamples_prediction, 3], dtype=np.float64)
    for (idx, dict_) in tq:
        datadict : dict[str,torch.Tensor] = dict_

        position_history = datadict["hist"]#[:,:,[0,1]]
        position_future = datadict["fut"]#[:,:,[0,1]]
        tangents_future = datadict["fut_tangents"]#[:,:,[0,1]]

        current_positions_full = datadict["current_position"]
        current_orientations_full = datadict["current_orientation"]

        rotations = Rotation.from_quat(current_orientations_full.detach().cpu().numpy())

        left_bound_input = datadict["left_bd"][:,:,[0,1]]
        right_bound_input = datadict["right_bd"][:,:,[0,1]]

        tracknames = datadict["trackname"]

        left_bound_label = datadict["future_left_bd"][:,:,[0,1]]
        right_bound_label = datadict["future_right_bd"][:,:,[0,1]]
        center_line_label = datadict["future_centerline"][:,:,[0,1]]
        optimal_line_label = datadict["future_raceline"][:,:,[0,1]]

        position_history = position_history.cuda(gpu_index).type(dtype)
        position_future = position_future.cuda(gpu_index).type(dtype)
        left_bound_input = left_bound_input.cuda(gpu_index).type(dtype)
        right_bound_input = right_bound_input.cuda(gpu_index).type(dtype)
        current_positions = current_positions_full.cuda(gpu_index).type(dtype)
        rotmats = torch.as_tensor(rotations.as_matrix()).cuda(gpu_index).type(dtype)
        tangents_future = tangents_future.cuda(gpu_index).type(dtype)
        tangents_future_global = torch.matmul(rotmats, tangents_future.unsqueeze(-1)).squeeze(-1)[:,:,[0,1]]
        tangents_future_global = tangents_future_global/torch.norm(tangents_future_global, p=2.0, dim=-1, keepdim=True)


        p0global = (torch.matmul(rotmats, position_future[:,0].unsqueeze(-1)).squeeze(-1) + current_positions)[:,[0,1]]
        position_future_global = torch.matmul(rotmats, position_future.transpose(-2,-1)).transpose(-2,-1)+current_positions[:,None]
        position_future_global = position_future_global[:,:,[0,1]]


        future_left_bd_array[idx,:] = datadict["future_left_bd"].cpu().numpy()[:]
        future_right_bd_array[idx,:] = datadict["future_right_bd"].cpu().numpy()[:]

        left_bound_label = left_bound_label.cuda(gpu_index).type(dtype)
        right_bound_label = right_bound_label.cuda(gpu_index).type(dtype)
        center_line_label = center_line_label.cuda(gpu_index).type(dtype)
        optimal_line_label = optimal_line_label.cuda(gpu_index).type(dtype)

        to_local_rotmats = rotmats.transpose(-2, -1)
        to_local_translations = torch.matmul(to_local_rotmats, -position_future[:,0].unsqueeze(-1)).squeeze(-1)

        currentbatchsize = int(position_history.shape[0])
        with torch.no_grad():

            # print(tangent_future[:,0])
            (mix_out_, init_speed_, acc_out_) = net(position_history[:,:,[0,1]], left_bound_input[:,:,[0,1]], right_bound_input[:,:,[0,1]])

            mix_out : torch.Tensor = mix_out_
            init_speed : torch.Tensor  = init_speed_
            acc_out : torch.Tensor = acc_out_ 
            relspeeds = torch.matmul(_time_profile_matrix, acc_out.T).T

            speed_out = torch.zeros_like(position_future[:,:,0])
            speed_out[:,0] = init_speed[:,0]
            speed_out[:,1:] = relspeeds + init_speed

            arclength_out = torch.zeros_like(speed_out)
            arclength_out[:,1:] = torch.cumsum( speed_out[:,:-1]*time_spacing, 1 )
            position_predicted_global = torch.zeros_like(position_future)#[:,:,[0,1]])

            for (i, trackname) in enumerate(tracknames):
                track = track_dict[trackname]
                mixed_path = torch.sum(track*mix_out[i,:,None,None], dim=0)
                mixed_path_deltas = mixed_path - p0global[i]
                mixed_path_delta_norms = torch.norm(mixed_path_deltas, p=2.0, dim=-1)
                Imin = torch.argmin(mixed_path_delta_norms)

                mixed_path_rolled = torch.roll(mixed_path, -(Imin.item() + 1), dims=0)
                mixed_path_arclengths = torch.zeros_like(mixed_path_rolled[:,0])
                mixed_path_arclengths[1:]+=torch.cumsum( torch.norm(mixed_path_rolled[1:] - mixed_path_rolled[:-1], p=2.0, dim=-1), 0)
                # print(mixed_path_rolled)
                # print(mixed_path_arclengths)
                # print(p0global[i])
                # ibucket = torch.bucketize(arclength_out, mixed_path_arclengths, right=True)
                dmat = torch.cdist(mixed_path_arclengths.unsqueeze(-1), arclength_out[i].unsqueeze(-1))
                Iclosest = torch.argmin(dmat, dim=0)

                position_predicted_global[i,:,[0,1]] = mixed_path_rolled[Iclosest]

                # print(dmat.shape)
                # print(Iclosest)
            position_predicted_local = torch.matmul(to_local_rotmats, position_predicted_global.unsqueeze(-1)).squeeze(-1)
            position_predicted_local += to_local_translations#[:,:,None]

            history_array[idx,:] = position_history[0].cpu().numpy()[:]
            prediction_array[idx,:] = position_predicted_local[0].cpu().numpy()[:]

            position_predicted_global = position_predicted_global[:,:,[0,1]]
                # exit(0)
            # batch_dim = position_future_global.shape[0]
            # ssamp_batch = ssamp.unsqueeze(0).expand(batch_dim, ssamp.shape[0]).cuda(gpu_index).type(dtype)
            normals_future_global = tangents_future_global[:,:,[1,0]].clone()
            normals_future_global[:,:,0]*=-1.0

            rotmats_decomposition = torch.stack([tangents_future_global, normals_future_global], axis=3).transpose(-2,-1)
            translations_decomposition = torch.matmul(rotmats_decomposition, -position_future_global.unsqueeze(-1)).squeeze(-1)

            decompositions = torch.matmul(rotmats_decomposition, position_predicted_global.unsqueeze(-1)).squeeze(-1) + translations_decomposition
            longitudinal_errors = torch.abs(decompositions[:,:,0])
            lateral_errors = torch.abs(decompositions[:,:,1])

            lateral_error_list+=torch.mean(lateral_errors, dim=1).cpu().numpy().tolist()
            longitudinal_error_list+=torch.mean(longitudinal_errors, dim=1).cpu().numpy().tolist()
            # print(rotmats_decomposition.shape)
            
            # kbezierfit=7
            # Mfuture, curves_future = deepracing_models.math_utils.bezierLsqfit(position_future_global, kbezierfit, t=ssamp_batch)
            # curves_deriv = kbezierfit*(curves_future[:,1:] - curves_future[:,:-1])/prediction_totaltime




            # positions_out = left_bound_label*mix_out[:,0,None,None] + right_bound_label*mix_out[:,1,None,None] + center_line_label*mix_out[:,2,None,None] + optimal_line_label*mix_out[:,3,None,None]
            # lateral_deltas = positions_out - position_future
            displacements = position_predicted_global - position_future_global

            de = torch.norm(displacements, p=2.0, dim=-1)
            ade = torch.mean(de, dim=-1)

            ade_list+=ade.cpu().numpy().tolist()






            # print("Lateral error: %f" % (torch.mean(lateral_error).item(),))

            

            # acc_out = torch.clamp(acc_out_ + speed_future[:,0].unsqueeze(-1), 5.0*one, 110.0*one)

        
    lateral_error_array = torch.as_tensor(lateral_error_list, dtype=torch.float64)
    longitudinal_error_array = torch.as_tensor(longitudinal_error_list, dtype=torch.float64)
    ade_array = torch.as_tensor(ade_list, dtype=torch.float64)
    results_dir = argdict["resultsdir"]
    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)
    with open(os.path.join(results_dir, "data.npz"), "wb") as f:
        np.savez(f, **{
            "lateral_error" : lateral_error_array.cpu().numpy(),
            "longitudinal_error" : longitudinal_error_array.cpu().numpy(),
            "ade" : ade_array.cpu().numpy(),
            "history" : history_array,
            "predictions" : prediction_array,
            "future_left_bd" : future_left_bd_array,
            "future_right_bd" : future_right_bd_array
        }
        )

    print("ADE: %f" % (torch.mean(ade_array).item(),))
    print("mean lateral error: %f" % (torch.mean(torch.abs(lateral_error_array)).item(),))
    print("mean longitudinal error: %f" % (torch.mean(torch.abs(longitudinal_error_array)).item(),))



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Test Bezier version of MixNet")
    parser.add_argument("--experiment", type=str, required=True, help="Which comet experiment to load")
    parser.add_argument("--tempdir", type=str, default="/bigtemp/ttw2xk/mixnet_bezier_dump", help="Temporary directory to save model files after downloading from comet.")
    parser.add_argument("--resultsdir", type=str, default="/p/DeepRacing/mixnet_results", help="Temporary directory to save model files after downloading from comet.")
    parser.add_argument("--workers", type=int, default=0, help="How many threads for data loading")
    args = parser.parse_args()
    argdict : dict = vars(args)
    argdict["batch_size"] = 1
    test(**argdict)