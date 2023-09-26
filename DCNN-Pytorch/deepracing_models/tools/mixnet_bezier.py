import argparse
import tempfile
import comet_ml
import deepracing_models.math_utils.bezier, deepracing_models.math_utils
from deepracing_models.nn_models.TrajectoryPrediction import BezierMixNet
from deepracing_models.data_loading import file_datasets as FD, SubsetFlag 
import torch, torch.optim, torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
import torch.utils.data as torchdata
import yaml
import os
import io
import numpy as np
import pickle as pkl
import tqdm
import matplotlib.figure
import matplotlib.pyplot as plt
import json
from scipy.spatial.transform import Rotation
import shutil
import glob
import multiprocessing, multiprocessing.pool
import traceback
import sys
from datetime import datetime
from threading import Semaphore, ThreadError
# k=3
# d = 2
# num = 1

def errorcb(exception):
    for elem in traceback.format_exception(exception):
        print(elem, flush=True, file=sys.stderr)

def load_datasets_from_shared_memory(
        shared_memory_locations : list[ tuple[ dict[str, tuple[str, list]], dict ]  ],
        dtype : np.dtype
    ):
    dsets : list[FD.TrajectoryPredictionDataset] = []
    for shm_dict, metadata_dict in shared_memory_locations:
        dsets.append(FD.TrajectoryPredictionDataset.from_shared_memory(shm_dict, metadata_dict, SubsetFlag.TRAIN, dtype=dtype))
    return dsets

def load_datasets_from_files(search_dir : str, kbezier : int, bcurve_cache = False):
    dsetfiles = glob.glob(os.path.join(search_dir, "**", "metadata.yaml"), recursive=True)
    dsets : list[FD.TrajectoryPredictionDataset] = []
    dsetconfigs = []
    numsamples_prediction = None
    for metadatafile in dsetfiles:
        with open(metadatafile, "r") as f:
            dsetconfig = yaml.load(f, Loader=yaml.SafeLoader)
        if numsamples_prediction is None:
            numsamples_prediction = dsetconfig["numsamples_prediction"]
        elif numsamples_prediction!=dsetconfig["numsamples_prediction"]:
            raise ValueError("All datasets must have the same number of prediction points. " + \
                            "Dataset at %s has prediction length %d, but previous dataset " + \
                            "has prediction length %d" % (metadatafile, dsetconfig["numsamples_prediction"], numsamples_prediction))
        dsetconfigs.append(dsetconfig)
        dsets.append(FD.TrajectoryPredictionDataset.from_file(metadatafile, SubsetFlag.TRAIN, dtype=np.float64))
        dsets[-1].fit_bezier_curves(kbezier, cache=bcurve_cache)
    return dsets

def train(allconfig : dict[str,dict] = None,
            tempdir : str = None,
            num_epochs : int = 200,
            workers : int = 0,
            shared_memory_keys : list[ tuple[ dict[str, tuple[str, list]], dict ]  ] | None = None,
            dtype : np.dtype | None = None,
            api_key : str | None = None):
    
    if allconfig is None:
        raise ValueError("keyword arg \"allconfig\" is mandatory")

    if tempdir is None:
        raise ValueError("keyword arg \"tempdir\" is mandatory")
    
    dataconfig = allconfig["data"]
    netconfig = allconfig["net"]
    trainerconfig = allconfig["trainer"]
    kbezier : int = trainerconfig["kbezier"]
    netconfig["kbezier"] = kbezier
    netconfig["gpu_index"] = trainerconfig["gpu_index"]
    constraint_first_deriv = netconfig["acc_decoder"]["constrain_derivatives"]["first"]
    constraint_second_deriv = netconfig["acc_decoder"]["constrain_derivatives"]["second"]
    if constraint_first_deriv ^ constraint_second_deriv:
        raise ValueError("Constraining one derivative but not both isn't done yet")
    fully_constrained = constraint_first_deriv and constraint_second_deriv
        
    project_name="mixnet-bezier"
    tags = allconfig["comet"]["tags"]
    if (api_key is not None):
        experiment = comet_ml.Experiment(workspace="electric-turtle", 
                                         project_name=project_name, 
                                         api_key=api_key, 
                                         auto_metric_logging=False, 
                                         auto_param_logging=False)
        for tag in tags:
            experiment.add_tag(tag)
    else:
        experiment = None
    if experiment is not None:
        tempdir_full = os.path.join(tempdir, experiment.name)
    else:
        tempdir_full = os.path.join(tempdir, "mixnet_bezier_" + datetime.now().strftime("%Y_%m_%d_%H:%M:%S"))

    if os.path.isdir(tempdir_full):
        shutil.rmtree(tempdir_full)

    os.makedirs(tempdir_full)

    if experiment is not None:
        config_copy = os.path.join(tempdir_full, "config_copy.yaml")
        with open(config_copy, "w") as f:
            yaml.dump(allconfig, f, Dumper=yaml.SafeDumper)
        experiment.log_asset(config_copy, "config.yaml", copy_to_tmp=False)
    else:
        os.mkdir(os.path.join(tempdir_full, "plots"))
    
    if shared_memory_keys is None:
        search_dir : str = dataconfig["dir"]
        datasets = load_datasets_from_files(search_dir, kbezier)
    else:
        if dtype is None:
            raise ValueError("If shared memory is specified, dtype must also be specified")
        datasets = load_datasets_from_shared_memory(shared_memory_keys, dtype)
    network : BezierMixNet = BezierMixNet(netconfig).double()
    lossfunc : torch.nn.MSELoss = torch.nn.MSELoss().double()
    gpu_index : int = trainerconfig["gpu_index"]
    use_cuda = gpu_index>=0
    if use_cuda:
        network = network.cuda(gpu_index)
        lossfunc = lossfunc.cuda(gpu_index)

    concat_dataset : torchdata.ConcatDataset = torchdata.ConcatDataset(datasets)
    dataloader : torchdata.DataLoader = torchdata.DataLoader(concat_dataset, batch_size=trainerconfig["batch_size"], pin_memory=use_cuda, shuffle=True, num_workers=workers)

    if type(num_epochs) is not int:
        raise ValueError("keyword arg \"num_epochs\" must be an int")
    
    if experiment is not None:
        print("Using comet. Experiment name: %s" % (experiment.get_name(),) )

    firstparam = next(network.parameters())
    dtype = firstparam.dtype
    device = firstparam.device
    lr = float(trainerconfig["learning_rate"])
    optimizername = trainerconfig["optimizer"]
    if optimizername=="SGD":
        momentum = trainerconfig["momentum"]
        nesterov = trainerconfig["nesterov"]
        optimizer = torch.optim.SGD(network.parameters(), lr = lr, momentum = momentum, nesterov=(nesterov and (momentum>0.0)))
    elif optimizername=="Adam":
        betas = tuple(trainerconfig["betas"])
        weight_decay = trainerconfig["weight_decay"]
        optimizer = torch.optim.Adam(network.parameters(), lr = lr, betas=betas, weight_decay = weight_decay)
    else:
        raise ValueError("Unknown optimizer %s" % (optimizername,))

    num_accel_sections : int = netconfig["acc_decoder"]["num_acc_sections"]
    prediction_totaltime = datasets[0].metadata["predictiontime"]
    network.train()
    averageloss = 1E9

    averagepositionloss = 1E9
    averagevelocityloss = 1E9
    averagearclengtherror = 1E9

    numsamples_prediction = datasets[0].metadata["numsamples_prediction"]
    tsamp = torch.linspace(0.0, prediction_totaltime, dtype=dtype, device=device, steps=numsamples_prediction)
    tswitchingpoints = torch.linspace(0.0, prediction_totaltime, dtype=dtype, device=device, steps=num_accel_sections+1)
    dt = tswitchingpoints[1:] - tswitchingpoints[:-1]
    kbeziervel = 3
    if experiment is not None:
        hyper_params_to_comet = {
            "kbezier" : kbezier,
            "kbezier_vel" : kbeziervel,
            "num_accel_sections" : num_accel_sections,
            "prediction_time" : prediction_totaltime,
            "time_switching_points" : tswitchingpoints.cpu().numpy().tolist(),
            "constrain_first_deriv" : constraint_first_deriv,
            "constraint_second_deriv" : constraint_second_deriv,
        }
        hyper_params_to_comet["tracks"] = list(set([dset.metadata["trackname"] for dset in datasets]))
        for dsetkey in ["numsamples_boundary", "numsamples_history", "numsamples_prediction", "predictiontime", "predictiontime", "historytime"]:
            hyper_params_to_comet[dsetkey] = datasets[0].metadata[dsetkey]
        experiment.log_parameters(hyper_params_to_comet)
    for epoch in range(1, trainerconfig["epochs"]+1):
        totalloss = 0.0
        total_position_loss = 0.0
        total_velocity_loss = 0.0
        total_arclength_error = 0.0
        total_ade = 0.0
        dataloader_enumerate = enumerate(dataloader)
        if experiment is None:
            tq = tqdm.tqdm(dataloader_enumerate, desc="Yay")
        else:
            tq = dataloader_enumerate
            experiment.set_epoch(epoch)
        if epoch%10==0:
            
            netout = os.path.join(tempdir_full, "network.pt")
            torch.save(network.state_dict(), netout)
            if experiment is not None:
                experiment.log_asset(netout, "network_epoch_%d.pt" % (epoch,), copy_to_tmp=False)   

            optimizerout =  os.path.join(tempdir_full, "optimizer.pt")
            torch.save(optimizer.state_dict(), optimizerout)
            if experiment is not None:
                experiment.log_asset(optimizerout, "optimizer_epoch_%d.pt" % (epoch,), copy_to_tmp=False)

        coordinate_idx_history = list(range(netconfig["input_dimension"]))
        coordinate_idx = list(range(2))
        for (i, dict_) in tq:
            datadict : dict[str,torch.Tensor] = dict_

            position_history = datadict["hist"][:,:,coordinate_idx_history].cuda(gpu_index).type(dtype)

            position_future = datadict["fut"].cuda(gpu_index).type(dtype)
            vel_future = datadict["fut_vel"].cuda(gpu_index).type(dtype)

            if coordinate_idx[-1]==2:
                future_arclength = datadict["future_arclength"].cuda(gpu_index).type(dtype)
            else:
                future_arclength = datadict["future_arclength_2d"].cuda(gpu_index).type(dtype)

            left_bound_input = datadict["left_bd"][:,:,coordinate_idx_history].cuda(gpu_index).type(dtype)
            right_bound_input = datadict["right_bd"][:,:,coordinate_idx_history].cuda(gpu_index).type(dtype)

            bcurves_r = datadict["reference_curves"].cuda(gpu_index).type(dtype)

            speed_future = torch.norm(vel_future[:,:,coordinate_idx], p=2.0, dim=-1)
            tangent_future = vel_future[:,:,coordinate_idx]/speed_future[:,:,None]

            currentbatchsize = int(position_history.shape[0])

            (mix_out_, acc_out_) = network(position_history, left_bound_input, right_bound_input)
            one = torch.ones_like(speed_future[0,0])
            mix_out = torch.clamp(mix_out_, -3.0*one, 3.0*one)
            # + speed_future[:,0].unsqueeze(-1)
            acc_out = acc_out_ + speed_future[:,0].unsqueeze(-1)
            # acc_out = torch.clamp(acc_out_ + speed_future[:,0].unsqueeze(-1), 5.0*one, 110.0*one)
            coefs_inferred = torch.zeros(currentbatchsize, num_accel_sections, kbeziervel+1, dtype=acc_out.dtype, device=acc_out.device)
            coefs_inferred[:,0,0] = speed_future[:,0]
            if fully_constrained:
                coefs_inferred[:,0,[1,2]] = acc_out[:,[0,1]]
                coefs_inferred[:,1:,1] = acc_out[:,2:-1]
                coefs_inferred[:,-1,-1] = acc_out[:,-1]
                for j in range(coefs_inferred.shape[1]-1):
                    coefs_inferred[:, j,-1] = coefs_inferred[:, j+1,0] = \
                        0.5*(coefs_inferred[:, j,-2] + coefs_inferred[:, j+1,1])
                    if kbeziervel>2:
                        coefs_inferred[:, j+1,-2] = 2.0*coefs_inferred[:, j+1,1] - 2.0*coefs_inferred[:, j, -2] + coefs_inferred[:, j, -3]
            else:
                setter_idx = [True, True, True, False]
                coefs_inferred[:,0,[1,2]] = acc_out[:,[0,1]]
                coefs_inferred[:,1:,setter_idx] = acc_out[:,2:-1].view(coefs_inferred[:,1:,setter_idx].shape)
                coefs_inferred[:,:-1,-1] = coefs_inferred[:,1:,0]   
                coefs_inferred[:,-1,-1] = acc_out[:,-1]  

            
            tstart_batch = tswitchingpoints[:-1].unsqueeze(0).expand(currentbatchsize, num_accel_sections)
            dt_batch = dt.unsqueeze(0).expand(currentbatchsize, num_accel_sections)
            teval_batch = tsamp.unsqueeze(0).expand(currentbatchsize, numsamples_prediction)


            speed_profile_out, idxbuckets = deepracing_models.math_utils.compositeBezierEval(tstart_batch, dt_batch, coefs_inferred.unsqueeze(-1), teval_batch)
            loss_velocity : torch.Tensor = (lossfunc(speed_profile_out, speed_future))#*(prediction_timestep**2)

            coefs_antiderivative = deepracing_models.math_utils.compositeBezierAntiderivative(coefs_inferred.unsqueeze(-1), dt_batch)
            arclengths_pred, _ = deepracing_models.math_utils.compositeBezierEval(tstart_batch, dt_batch, coefs_antiderivative, teval_batch, idxbuckets=idxbuckets)
            delta_r_pred = arclengths_pred[:,-1]
            s_pred : torch.Tensor = arclengths_pred/delta_r_pred[:,None]

            delta_r_gt : torch.Tensor = future_arclength[:,-1] - future_arclength[:,0]
            future_arclength_rel : torch.Tensor = future_arclength - future_arclength[:,0,None]
            s_gt : torch.Tensor = future_arclength_rel/delta_r_gt[:,None]

            known_control_points : torch.Tensor = torch.zeros_like(bcurves_r[:,0,:2,coordinate_idx])
            mixed_control_points = torch.sum(bcurves_r[:,:,:,coordinate_idx]*mix_out[:,:,None,None], dim=1)
            mcp_deltar : torch.Tensor = deepracing_models.math_utils.bezierArcLength(mixed_control_points, num_segments = 10)

            # known_control_points[:,0] = position_future[:,0,coordinate_idx]
            # known_control_points[:,0] + 
            known_control_points[:,1] = (delta_r_gt[:,None]/kbezier)*tangent_future[:,0,coordinate_idx]
            predicted_bcurve = torch.cat([known_control_points, mixed_control_points[:,2:]], dim=1) 
            pred_deltar : torch.Tensor = deepracing_models.math_utils.bezierArcLength(predicted_bcurve, num_segments = 10)

            arclengths_gt_s = future_arclength_rel/delta_r_gt[:,None]
            Mbezier_gt = deepracing_models.math_utils.bezierM(arclengths_gt_s, kbezier)

            loss_arclength : torch.Tensor = lossfunc(arclengths_pred, future_arclength_rel)
            
            arclengths_pred_s = (arclengths_pred/delta_r_gt[:,None]).clamp(min=0.0, max=1.0)

            Marclengths_pred : torch.Tensor = deepracing_models.math_utils.bezierM(arclengths_pred_s, kbezier)
            pointsout : torch.Tensor = torch.matmul(Marclengths_pred, predicted_bcurve)
            displacements : torch.Tensor = pointsout - position_future[:,:,coordinate_idx]
            ade : torch.Tensor = torch.mean(torch.norm(displacements, p=2.0, dim=-1))

            predicted_position_lateral_only = torch.matmul(Mbezier_gt, predicted_bcurve)

            lateral_error : torch.Tensor = lossfunc(predicted_position_lateral_only, position_future[:,:,coordinate_idx])
            if experiment is not None:
                experiment.log_metric("lateral_error", lateral_error.item())
                experiment.log_metric("loss_arclength", loss_arclength.item())
                experiment.log_metric("mean_displacement_error", ade.item())
                experiment.log_metric("loss_velocity", loss_velocity.item())
            if trainerconfig["ade_loss"] and (not torch.isnan(ade)) and ade<1000.0:     
                loss = ade
            else:
                loss = lateral_error + loss_velocity

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if experiment is None:
                total_position_loss += lateral_error.item()
                total_velocity_loss += loss_velocity.item()
                total_arclength_error += loss_arclength.item()
                total_ade += ade.item()
                totalloss += loss.item()
                averageloss = totalloss/(i+1)
                averagepositionloss = total_position_loss/(i+1)
                averagevelocityloss = total_velocity_loss/(i+1)
                averagearclengtherror = total_arclength_error/(i+1)
                averageade = total_ade/(i+1)
                tq.set_postfix({"average position loss" : averagepositionloss, 
                                "average velocity loss" : averagevelocityloss, 
                                "average arclength loss" : averagearclengtherror, 
                                "average loss" : averageloss, 
                                "average ade" : averageade, 
                                "epoch": epoch})

        if epoch%10==0:
            bcurves_r_cpu = bcurves_r[0].cpu()
            position_history_cpu = position_history[0].cpu()
            position_future_cpu = position_future[0].cpu()
            predicted_position_future_cpu = predicted_position_lateral_only[0].detach().cpu()
            left_bound_label_cpu = datadict["future_left_bd"][0].cpu()
            right_bound_label_cpu = datadict["future_right_bd"][0].cpu()
            centerline_label_cpu = datadict["future_centerline"][0].cpu()
            raceline_label_cpu = datadict["future_raceline"][0].cpu()
            fig : matplotlib.figure.Figure = plt.figure()
            scale_array = 5.0
            plt.plot(position_history_cpu[:,0], position_history_cpu[:,1], label="Position History")#, s=scale_array)
            plt.plot(position_history_cpu[0,0], position_history_cpu[0,1], "g*", label="Position History Start")
            plt.plot(position_history_cpu[-1,0], position_history_cpu[-1,1], "r*", label="Position History End")
            plt.scatter(position_future_cpu[:,0], position_future_cpu[:,1], label="Ground Truth Future", s=scale_array)
            plt.plot(predicted_position_future_cpu[:,0], predicted_position_future_cpu[:,1], label="Prediction")#, s=scale_array)
            plt.plot(centerline_label_cpu[:,0], centerline_label_cpu[:,1], label="Centerline Label")#, s=scale_array)
            plt.plot(raceline_label_cpu[:,0], raceline_label_cpu[:,1], label="Raceline Label")#, s=scale_array)
            plt.plot([],[], label="Boundaries", color="navy")#, s=scale_array)
            plt.legend()
            plt.plot(left_bound_label_cpu[:,0], left_bound_label_cpu[:,1], label="Left Bound Label", color="navy")#, s=scale_array)
            plt.plot(right_bound_label_cpu[:,0], right_bound_label_cpu[:,1], label="Right Bound Label", color="navy")#, s=scale_array)
            plt.axis("equal")
            fig_velocity : matplotlib.figure.Figure = plt.figure()
            tsamp_cpu : np.ndarray = tsamp.cpu().numpy()
            plt.plot(tsamp_cpu, speed_future[0].cpu().numpy(), linestyle="dashed", label="Ground Truth Speed")
            plt.plot(tsamp_cpu, speed_profile_out[0].detach().cpu().numpy(), label="Predicted Speed")
            all_speeds = torch.cat([speed_future[0], speed_profile_out[0]], dim=0)
            plt.vlines(tswitchingpoints.cpu().numpy(), all_speeds.min().item() - 1.0, all_speeds.max().item() + 1.0,\
                       linestyle="dashed", color="grey", alpha=0.5)
            plt.xlabel("$t$", usetex=True)
            plt.ylabel("$\\nu(t)$", usetex=True)
            plt.tight_layout(pad=0.75)
            plt.gca().yaxis.label.set(rotation='horizontal', ha='right')
            if experiment is not None:
                experiment.log_figure(figure_name="positions_epoch_%d" % (epoch,), figure=fig)
                experiment.log_figure(figure_name="speeds_epoch_%d" % (epoch,), figure=fig_velocity)
            else:
                fig.savefig(os.path.join(tempdir_full, "plots", "positions_epoch_%d.pdf" % (epoch,)))
                fig_velocity.savefig(os.path.join(tempdir_full, "plots", "speeds_epoch_%d.pdf" % (epoch,)))
            plt.close(fig=fig)
            plt.close(fig=fig_velocity)

def prepare_and_train(argdict : dict):
    tempdir = argdict["tempdir"]
    workers = argdict["workers"]
    config_file = argdict["config_file"]
    with open(config_file, "r") as f:
        allconfig : dict = yaml.load(f, Loader=yaml.SafeLoader)
    train(allconfig=allconfig, workers=workers, tempdir=tempdir, api_key=os.getenv("COMET_API_KEY"))

    
            
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train Bezier version of MixNet")
    parser.add_argument("config_file", type=str,  help="Configuration file to load")
    parser.add_argument("--tempdir", type=str, required=True, help="Temporary directory to save model files before uploading to comet. Default is to use tempfile module to generate one")
    parser.add_argument("--workers", type=int, default=0, help="How many threads for data loading")
    args = parser.parse_args()
    argdict : dict = vars(args)
    prepare_and_train(argdict)