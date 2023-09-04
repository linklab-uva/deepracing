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
def loadwrapper(dsets : list[FD.TrajectoryPredictionDataset], dsetconfigs : list[dict], metadatafile : str, kbezier : int, mutex : Semaphore, built_in_lstq=True):
    with open(metadatafile, "r") as f:
        dsetconfig = yaml.load(f, Loader=yaml.SafeLoader)
    dset = FD.TrajectoryPredictionDataset(metadatafile, SubsetFlag.TRAIN, dtype=torch.float64)
    dset.fit_bezier_curves(kbezier, built_in_lstq=built_in_lstq)
    if mutex.acquire(blocking=True, timeout=2.0):
        dsets.append(dset)
        dsetconfigs.append(dsetconfig)
        mutex.release()
    else:
        raise RuntimeError("Could not acquire mutex for dataset: %s" % (metadatafile,))
def trainmixnet(argdict : dict):
    config_file = argdict["config_file"]
    project_name="mixnet-bezier"
    api_key = os.getenv("COMET_API_KEY")
    tempdir = argdict["tempdir"]
    if (api_key is not None) and (not argdict["offline"]):
        experiment = comet_ml.Experiment(workspace="electric-turtle", project_name=project_name, api_key=api_key)
        print(api_key)
    elif (api_key is not None) and (tempdir is not None) and argdict["offline"]:
        offline_name = "mixnet_bezier_" + datetime.now().strftime("%Y_%m_%d_%H:%M:%S") 
        experiment = comet_ml.OfflineExperiment(workspace="electric-turtle", project_name=project_name, api_key=api_key,\
                                                offline_directory=tempdir)
        experiment.set_name(offline_name)
    else:
        experiment = None
    if tempdir is None:
        tempdirobj = tempfile.TemporaryDirectory()
        tempdir_full = tempdirobj.name
    else:
        if experiment is not None:
            tempdir_full = os.path.join(tempdir, experiment.name)
        else:
            tempdir_full = os.path.join(tempdir, datetime.now().strftime("%Y_%m_%d_%H:%M:%S"))
        if os.path.isdir(tempdir_full):
            shutil.rmtree(tempdir_full)
        os.makedirs(tempdir_full)
    if experiment is not None:
        shutil.copy(config_file, tempdir_full)
        experiment.log_asset(os.path.join(tempdir_full, os.path.basename(config_file)), "config.yaml", copy_to_tmp=False)
    else:
        os.mkdir(os.path.join(tempdir_full, "plots"))
    with open(config_file, "r") as f:
        allconfig : dict = yaml.load(f, Loader=yaml.SafeLoader)
    dataconfig = allconfig["data"]
    netconfig = allconfig["net"]
    trainerconfig = allconfig["trainer"]
    gpu_index=trainerconfig["gpu_index"]
    cuda=gpu_index>=0

    datadir = dataconfig["dir"]
    dsetfiles = glob.glob(os.path.join(datadir, "**", "metadata.yaml"), recursive=True)
    dsets : list[FD.TrajectoryPredictionDataset] = []
    mutex : Semaphore = Semaphore()
    dsetconfigs = []
    asyncresults = []
    numsamples_prediction = None
    kbezier = trainerconfig["kbezier"]
    # if cuda:
    #     dev = torch.device("cuda:%d" % (gpu_index,))
    # else:
    #     dev = torch.device("cpu")
    # with multiprocessing.pool.Pool(processes=argdict["threads"]) as threadpool:
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
        dsets.append(FD.TrajectoryPredictionDataset(metadatafile, SubsetFlag.TRAIN, dtype=torch.float64))
        dsets[-1].fit_bezier_curves(kbezier, built_in_lstq=True, cache=True)
        #def loadwrapper(dsets : list[FD.TrajectoryPredictionDataset], dsetconfigs : list[dict], metadatafile : str, kbezier : int, mutex : Semaphore, built_in_lstq=True):
        # loadargs=[dsets, dsetconfigs, metadatafile, kbezier, mutex]
        # loadkwds = {"built_in_lstq" : True}
        # asyncresults.append(threadpool.apply_async(loadwrapper, args=loadargs, kwds=loadkwds, error_callback=errorcb))
        # threadpool.close()
        # threadpool.join()   
    batch_size = trainerconfig["batch_size"]
    netconfig["gpu_index"]=gpu_index
    dataloader = torchdata.DataLoader(torchdata.ConcatDataset(dsets), batch_size=batch_size, pin_memory=cuda, shuffle=True)#, collate_fn=dsets[0].collate_fn)
    net : BezierMixNet = BezierMixNet(netconfig).double()
    lossfunc = torch.nn.MSELoss().double()
    if cuda:
        net = net.cuda(gpu_index)
        lossfunc = lossfunc.cuda(gpu_index)
    firstparam = next(net.parameters())
    dtype = firstparam.dtype
    device = firstparam.device
    nesterov = trainerconfig["nesterov"]
    lr = float(trainerconfig["learning_rate"])
    betas = tuple(trainerconfig["betas"])
    weight_decay = trainerconfig["weight_decay"]
    optimizername = trainerconfig["optimizer"]
    if optimizername=="SGD":
        momentum = trainerconfig["momentum"]
        optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = momentum, nesterov=(nesterov and (momentum>0.0)))
    elif optimizername=="Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr = lr, betas=betas, weight_decay = weight_decay)
    else:
        raise ValueError("Unknown optimizer %s" % (optimizername,))

    num_accel_sections : int = netconfig["acc_decoder"]["num_acc_sections"]
    prediction_timestep = dsetconfigs[0]["timestep_prediction"]
    prediction_totaltime = dsetconfigs[0]["predictiontime"]
    net.train()
    averageloss = 1E9
    averagepositionloss = 1E9
    averagevelocityloss = 1E9
    averagevelocityerror = 1E9
    averagearclengtherror = 1E9

    numsamples_prediction = dsetconfigs[0]["numsamples_prediction"]
    tsamp = torch.linspace(0.0, prediction_totaltime, dtype=dtype, device=device, steps=numsamples_prediction)
    tswitchingpoints = torch.linspace(0.0, prediction_totaltime, dtype=dtype, device=device, steps=num_accel_sections+1)
    dt = tswitchingpoints[1:] - tswitchingpoints[:-1]
    kbeziervel = 3
    for epoch in range(1, trainerconfig["epochs"]+1):
        totalloss = 0.0
        total_position_loss = 0.0
        total_velocity_loss = 0.0
        total_velocity_error = 0.0
        total_arclength_error = 0.0
        dataloader_enumerate = enumerate(dataloader)
        if experiment is None:
            tq = tqdm.tqdm(dataloader_enumerate, desc="Yay")
        else:
            tq = dataloader_enumerate
        if epoch%10==0:
            
            netout = os.path.join(tempdir_full, "net.pt")
            torch.save(net.state_dict(), netout)
            if experiment is not None:
                experiment.log_asset(netout, "network_epoch_%d.pt" % (epoch,), copy_to_tmp=False)   

            optimizerout =  os.path.join(tempdir_full, "optimizer.pt")
            torch.save(optimizer.state_dict(), optimizerout)
            if experiment is not None:
                experiment.log_asset(optimizerout, "optimizer_epoch_%d.pt" % (epoch,), copy_to_tmp=False)
        for (i, dict_) in tq:
            datadict : dict[str,torch.Tensor] = dict_

            position_history = datadict["hist"]
            position_future = datadict["fut"]
            tangent_future = datadict["fut_tangents"]
            speed_future = datadict["fut_speed"]
            future_arclength = datadict["future_arclength"]

            left_bound_input = datadict["left_bd"]
            right_bound_input = datadict["right_bd"]

            bcurves_r = datadict["reference_curves"]

            if cuda:
                position_history = position_history.cuda(gpu_index).type(dtype)
                position_future = position_future.cuda(gpu_index).type(dtype)
                speed_future = speed_future.cuda(gpu_index).type(dtype)
                left_bound_input = left_bound_input.cuda(gpu_index).type(dtype)
                right_bound_input = right_bound_input.cuda(gpu_index).type(dtype)
                future_arclength = future_arclength.cuda(gpu_index).type(dtype)
                bcurves_r = bcurves_r.cuda(gpu_index).type(dtype)
                tangent_future = tangent_future.cuda(gpu_index).type(dtype)
            else:
                position_history = position_history.cpu().type(dtype)
                position_future = position_future.cpu().type(dtype)
                speed_future = speed_future.cpu().type(dtype)
                left_bound_input = left_bound_input.cpu().type(dtype)
                right_bound_input = right_bound_input.cpu().type(dtype)
                future_arclength = future_arclength.cpu().type(dtype)
                bcurves_r = bcurves_r.cpu().type(dtype)
                tangent_future = tangent_future.cpu().type(dtype)

            currentbatchsize = int(position_history.shape[0])


            # print(tangent_future[:,0])
            (mix_out_, acc_out_) = net(position_history[:,:,[0,1]], left_bound_input[:,:,[0,1]], right_bound_input[:,:,[0,1]])
            one = torch.ones_like(speed_future[0,0])
            mix_out = torch.clamp(mix_out_, -0.5*one, 1.5*one)
            # + speed_future[:,0].unsqueeze(-1)
            acc_out = acc_out_ + speed_future[:,0].unsqueeze(-1)
            # acc_out = torch.clamp(acc_out_ + speed_future[:,0].unsqueeze(-1), 5.0*one, 110.0*one)
            

            coefs_inferred = torch.zeros(currentbatchsize, num_accel_sections, 4, dtype=acc_out.dtype, device=acc_out.device)
            coefs_inferred[:,0,0] = speed_future[:,0]
            coefs_inferred[:,0,[1,2]] = acc_out[:,[0,1]]
            coefs_inferred[:,1:,1] = acc_out[:,2:-1]
            coefs_inferred[:,-1,-1] = acc_out[:,-1]
            for j in range(coefs_inferred.shape[1]-1):
                coefs_inferred[:, j,-1] = coefs_inferred[:, j+1,0] = \
                    0.5*(coefs_inferred[:, j,-2] + coefs_inferred[:, j+1,1])
                if kbeziervel>2:
                    coefs_inferred[:, j+1,-2] = 2.0*coefs_inferred[:, j+1,1] - 2.0*coefs_inferred[:, j, -2] + coefs_inferred[:, j, -3]
            tstart_batch = tswitchingpoints[:-1].unsqueeze(0).expand(currentbatchsize, num_accel_sections)
            dt_batch = dt.unsqueeze(0).expand(currentbatchsize, num_accel_sections)
            teval_batch = tsamp.unsqueeze(0).expand(currentbatchsize, numsamples_prediction)
            speed_profile_out, idxbuckets = deepracing_models.math_utils.compositeBezierEval(tstart_batch, dt_batch, coefs_inferred.unsqueeze(-1), teval_batch)
            loss_velocity : torch.Tensor = (lossfunc(speed_profile_out, speed_future))*(prediction_timestep**2)
            if experiment is not None:
                experiment.log_metric("loss_velocity", loss_velocity.item())
            

            delta_r : torch.Tensor = future_arclength[:,-1] - future_arclength[:,0]
            future_arclength_rel : torch.Tensor = future_arclength - future_arclength[:,0,None]
            arclengths_out_s = future_arclength_rel/delta_r[:,None]
            known_control_points : torch.Tensor = torch.zeros_like(bcurves_r[:,0,:2])
            known_control_points[:,0] = position_future[:,0]
            known_control_points[:,1] = known_control_points[:,0] + (delta_r[:,None]/kbezier)*tangent_future[:,0]
            mixed_control_points = torch.sum(bcurves_r[:,:,2:]*mix_out[:,:,None,None], dim=1)
            predicted_bcurve = torch.cat([known_control_points, mixed_control_points], dim=1) 

            coefs_antiderivative = deepracing_models.math_utils.compositeBezierAntiderivative(coefs_inferred.unsqueeze(-1), dt_batch)
            arclengths_pred, _ = deepracing_models.math_utils.compositeBezierEval(tstart_batch, dt_batch, coefs_antiderivative, teval_batch, idxbuckets=idxbuckets)
            loss_arclength : torch.Tensor = lossfunc(arclengths_pred, future_arclength_rel)
            arclengths_pred_s = arclengths_pred/arclengths_pred[:,-1,None]
            Marclengths_out : torch.Tensor = deepracing_models.math_utils.bezierM(arclengths_pred_s, kbezier)
            pointsout : torch.Tensor = torch.matmul(Marclengths_out, predicted_bcurve)
            displacements : torch.Tensor = pointsout - position_future
            ade : torch.Tensor = torch.mean(torch.norm(displacements, p=2.0, dim=-1))
            if experiment is not None:
                experiment.log_metric("loss_arclength", loss_arclength.item())
                experiment.log_metric("mean_displacement_error", ade.item())

            Mbezierout = deepracing_models.math_utils.bezierM(arclengths_out_s, kbezier)
            predicted_position_future = torch.matmul(Mbezierout, predicted_bcurve)

            loss_position : torch.Tensor = lossfunc(predicted_position_future, position_future)
            if experiment is not None:
                experiment.log_metric("loss_position", loss_position.item())
                experiment.log_metric("velocity_error", loss_velocity.item()/(prediction_timestep**2))
            

            loss = loss_position + 2.0*loss_velocity #+ 10.0*loss_arclength #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if experiment is None:
                total_position_loss += loss_position.item()
                total_velocity_loss += loss_velocity.item()
                total_velocity_error += loss_velocity.item()/(prediction_timestep**2)
                total_arclength_error += loss_arclength.item()
                totalloss += loss.item()
                averageloss = totalloss/(i+1)
                averagepositionloss = total_position_loss/(i+1)
                averagevelocityloss = total_velocity_loss/(i+1)
                averagevelocityerror = total_velocity_error/(i+1)
                averagearclengtherror = total_arclength_error/(i+1)
                tq.set_postfix({"average position loss" : averagepositionloss, 
                                "average velocity error" : averagevelocityerror, 
                                "average velocity loss" : averagevelocityloss, 
                                "average arclength loss" : averagearclengtherror, 
                                "average loss" : averageloss, 
                                "epoch": epoch})

        if epoch%10==0:
            bcurves_r_cpu = bcurves_r[0].cpu()
            position_history_cpu = position_history[0].cpu()
            position_future_cpu = position_future[0].cpu()
            predicted_position_future_cpu = predicted_position_future[0].detach().cpu()
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
            
            
            # plt.show()
            # plt.show()
        # scheduler.step()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train Bezier version of MixNet")
    parser.add_argument("config_file", type=str,  help="Configuration file to load")
    parser.add_argument("--tempdir", type=str, default=None, help="Temporary directory to save model files before uploading to comet. Default is to use tempfile module to generate one")
    parser.add_argument("--threads", type=int, default=1, help="How many threads for pre-processing bcurves")
    parser.add_argument("--offline", action="store_true", help="Run as an offline comet experiment instead of uploading to comet.ml")
    args = parser.parse_args()
    argdict : dict = vars(args)
    trainmixnet(argdict)