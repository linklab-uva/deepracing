import argparse
import comet_ml
import deepracing_models.math_utils.bezier, deepracing_models.math_utils
from deepracing_models.nn_models.TrajectoryPrediction import BezierMixNet
from deepracing_models.data_loading import file_datasets as FD, SubsetFlag 
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
    api_experiment : comet_ml.APIExperiment = api.get(workspace="electric-turtle", project_name="mixnet-bezier", experiment=experiment)
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
    net_assets = sorted(net_assets, key=assetkey)
    optimizer_assets = sorted(optimizer_assets, key=assetkey)
    

    config_str = str(api_experiment.get_asset(config_asset["assetId"]), encoding="ascii")
    # print(config_str)
    config = yaml.safe_load(config_str)
    print(config)
    if not os.path.isdir(experiment_dir):
        raise ValueError("PANIK!!!!!1!!!!ONEONE!!!!!")
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)
    net_file = os.path.join(experiment_dir, "net.pt")
    if not os.path.isfile(net_file):
        net_asset = api_experiment.get_asset(net_assets[-1]["assetId"], return_type="binary")
        with open(net_file, "wb") as f:
            f.write(net_asset)
    netconfig = config["net"]
    netconfig["gpu_index"]=gpu_index
    device = torch.device("cuda:%d" % (config["net"]["gpu_index"],))
    net : BezierMixNet = BezierMixNet(config["net"]).double().eval()
    with open(net_file, "rb") as f:
        state_dict = torch.load(f, map_location=device)
    net.load_state_dict(state_dict)
    firstparam = next(net.parameters())
    dtype = firstparam.dtype
    data_config = config["data"]
    datadir = data_config["dir"]
    dsetfiles = glob.glob(os.path.join(datadir, "**", "metadata.yaml"), recursive=True)
    numsamples_prediction = None
    trainerconfig = config["trainer"]
    kbezier = trainerconfig["kbezier"]
    num_accel_sections : int = netconfig["acc_decoder"]["num_acc_sections"]
    dsetconfigs : list[dict] = []
    dsets : list[FD.TrajectoryPredictionDataset] = []
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
        direct_load = True
        dsets.append(FD.TrajectoryPredictionDataset(metadatafile, SubsetFlag.VAL, direct_load, dtype=torch.float64))
        bcurve_cache = False
        dsets[-1].fit_bezier_curves(kbezier, built_in_lstq=True, cache=bcurve_cache)
    numsamples_prediction = dsetconfigs[0]["numsamples_prediction"]
    prediction_totaltime = dsetconfigs[0]["predictiontime"]

    tsamp = torch.linspace(0.0, prediction_totaltime, dtype=dtype, device=device, steps=numsamples_prediction)
    tswitchingpoints = torch.linspace(0.0, prediction_totaltime, dtype=dtype, device=device, steps=num_accel_sections+1)
    dt = tswitchingpoints[1:] - tswitchingpoints[:-1]
    kbeziervel = 3


    concat_set = torchdata.ConcatDataset(dsets)
    num_samples = len(concat_set)
    dataloader = torchdata.DataLoader(concat_set, num_workers=workers, batch_size=batch_size, pin_memory=True, shuffle=False)
    dataloader_enumerate = enumerate(dataloader)
    tq = tqdm.tqdm(dataloader_enumerate, desc="Yay", total=int(np.ceil(num_samples/batch_size)))
    ade_list = []
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

        position_history = position_history.cuda(gpu_index).type(dtype)
        position_future = position_future.cuda(gpu_index).type(dtype)
        speed_future = speed_future.cuda(gpu_index).type(dtype)
        left_bound_input = left_bound_input.cuda(gpu_index).type(dtype)
        right_bound_input = right_bound_input.cuda(gpu_index).type(dtype)
        future_arclength = future_arclength.cuda(gpu_index).type(dtype)
        bcurves_r = bcurves_r.cuda(gpu_index).type(dtype)
        tangent_future = tangent_future.cuda(gpu_index).type(dtype)

        currentbatchsize = int(position_history.shape[0])
        with torch.no_grad():

            # print(tangent_future[:,0])
            (mix_out_, acc_out_) = net(position_history[:,:,[0,1]], left_bound_input[:,:,[0,1]], right_bound_input[:,:,[0,1]])
            one = torch.ones_like(speed_future[0,0])
            mix_out = torch.clamp(mix_out_, -0.5*one, 1.5*one)
            # + speed_future[:,0].unsqueeze(-1)
            acc_out = acc_out_ + speed_future[:,0].unsqueeze(-1)
            # acc_out = torch.clamp(acc_out_ + speed_future[:,0].unsqueeze(-1), 5.0*one, 110.0*one)
            

            coefs_inferred = torch.zeros(currentbatchsize, num_accel_sections, kbeziervel+1, dtype=acc_out.dtype, device=acc_out.device)
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
            


            coefs_antiderivative = deepracing_models.math_utils.compositeBezierAntiderivative(coefs_inferred.unsqueeze(-1), dt_batch)
            arclengths_pred, _ = deepracing_models.math_utils.compositeBezierEval(tstart_batch, dt_batch, coefs_antiderivative, teval_batch, idxbuckets=idxbuckets)
            arclengths_pred_s = arclengths_pred/arclengths_pred[:,-1,None]
            Marclengths_pred : torch.Tensor = deepracing_models.math_utils.bezierM(arclengths_pred_s, kbezier)
            mixed_control_points = torch.sum(bcurves_r[:,:,:,[0,1]]*mix_out[:,:,None,None], dim=1)

            # delta_r : torch.Tensor = arclengths_pred[:,-1] - arclengths_pred[:,0]
            # delta_r : torch.Tensor = future_arclength[:,-1] - future_arclength[:,0]
            delta_r : torch.Tensor = deepracing_models.math_utils.bezierArcLength(mixed_control_points, quadrature_order=9)


            known_control_points : torch.Tensor = torch.zeros_like(bcurves_r[:,0,:2,[0,1]])
            known_control_points[:,0] = position_future[:,0,[0,1]]
            known_control_points[:,1] = known_control_points[:,0] + (delta_r[:,None]/kbezier)*tangent_future[:,0,[0,1]]

            predicted_bcurve = torch.cat([known_control_points, mixed_control_points[:,2:]], dim=1) 

            pointsout : torch.Tensor = torch.matmul(Marclengths_pred, predicted_bcurve)
            displacements : torch.Tensor = pointsout[:,:,[0,1]] - position_future[:,:,[0,1]]
            displacement_norms = torch.norm(displacements, p=2.0, dim=-1)
            ade : torch.Tensor = torch.mean(displacement_norms, dim=-1)
            ade_list+=ade.cpu().numpy().tolist()
        
    ade_array = torch.as_tensor(ade_list, dtype=torch.float64)
    print(torch.mean(ade_array))



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Test Bezier version of MixNet")
    parser.add_argument("--experiment", type=str, required=True, help="Which comet experiment to load")
    parser.add_argument("--tempdir", type=str, default="/bigtemp/ttw2xk/mixnet_bezier_dump", help="Temporary directory to save model files after downloading from comet.")
    parser.add_argument("--workers", type=int, default=0, help="How many threads for data loading")
    args = parser.parse_args()
    argdict : dict = vars(args)
    argdict["batch_size"] = 128
    test(**argdict)