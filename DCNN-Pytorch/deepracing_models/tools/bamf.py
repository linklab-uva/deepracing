import argparse
import comet_ml
from deepracing_models.data_loading import file_datasets as FD, SubsetFlag 
from deepracing_models.data_loading.utils.file_utils import load_datasets_from_files
from deepracing_models.nn_models.trajectory_prediction import BAMF
import torch, torch.nn, torch.utils.data as torchdata
import yaml
import os
import numpy as np
import traceback
import sys
import deepracing_models.math_utils
import tqdm
# t = torch.linspace(0.0, 3.0, steps=num_segments+1).to(device=device, dtype=dtype).unsqueeze(0).expand(batchdim, num_segments+1)
# tstart = t[:,:-1]
# dt = t[:,1:] - tstart
# nhistory = 61
# nsamp = 61
# tsamp = torch.linspace(0.0, 3.0, steps=nsamp).to(device=device, dtype=dtype).unsqueeze(0).expand(batchdim, nsamp).clone()
# random_history = torch.randn(batchdim, nhistory, 4).to(device=device, dtype=dtype)
# p0 = random_history[:,-1,:2]
# v0 = random_history[:,-1,2:]
# random_lb = torch.randn(batchdim, nsamp, 4).to(device=device, dtype=dtype)
# random_rb = torch.randn(batchdim, nsamp, 4).to(device=device, dtype=dtype)
# random_gt = torch.randn(batchdim, nsamp, 2).to(device=device, dtype=dtype)
# print("here we go")
# tick = time.time()
# velcurveout, poscurveout = network(random_history, random_lb, random_rb, dt, v0)#, p0=p0)
# pout, idxbuckets = deepracing_models.math_utils.compositeBezierEval(tstart, dt, poscurveout, tsamp)
# fakedeltas = pout - random_gt
# fakeloss = torch.mean(torch.norm(fakedeltas, dim=-1))
# fakeloss.backward()
# tock = time.time()
# print(velcurveout[0])
# print(poscurveout[0])
# print("done. took %f seconds" % (tock-tick,))
# print(tsamp.shape)
# print(poscurveout.shape)

def errorcb(exception):
    for elem in traceback.format_exception(exception):
        print(elem, flush=True, file=sys.stderr)

def train(config : dict = None, tempdir : str = None, num_epochs : int = 200, 
          workers : int = 0, comet_experiment : comet_ml.Experiment | None = None, gpu : int = -1):

    if config is None:
        raise ValueError("keyword arg \"config\" is mandatory")
    if tempdir is None:
        raise ValueError("keyword arg \"tempdir\" is mandatory")
    if comet_experiment is not None:
        tempdir = os.path.join(tempdir, comet_experiment.get_name())
    else:
        tempdir = os.path.join(tempdir, "debug")
    os.makedirs(tempdir, exist_ok=True)
    netconfig = config["network"]
    num_segments = netconfig["num_segments"]
    kbezier = netconfig["kbezier"]
    with_batchnorm = netconfig["with_batchnorm"]
    if netconfig.get("heading_encoding", "quaternion")=="angle":
        print("Using heading angle as orientation input")
        history_dimension = 5
        heading_input_quaternion = False
    else:
        print("Using quaternion as orientation input")
        history_dimension = 6
        heading_input_quaternion = True
    network : BAMF = BAMF( history_dimension = history_dimension,
            num_segments = num_segments, 
            kbezier = kbezier,
            with_batchnorm = with_batchnorm
        ).train()
    
    trainerconfig = config["trainer"]
    if trainerconfig["float32"]:
        network = network.float()
    else:
        network = network.double()
    if gpu>=0:
        network = network.cuda(gpu)

    firstparam = next(network.parameters())
    device = firstparam.device
    dtype = firstparam.dtype
    lr = float(trainerconfig["learning_rate"])
    betas = tuple(trainerconfig["betas"])
    weight_decay = trainerconfig["weight_decay"]
    optimizer = torch.optim.Adam(network.parameters(), lr = lr, betas=betas, weight_decay = weight_decay)
    dataconfig = config["data"]
    search_dirs = dataconfig["dirs"]
    keys : set = {
        "hist",
        "hist_quats",
        "hist_vel",
        "fut",
        "fut_vel",
        "left_bd",
        "left_bd_tangents",
        "right_bd",
        "right_bd_tangents",
    }
    datasets : list[FD.TrajectoryPredictionDataset] = []
    for search_dir in search_dirs:
        datasets += load_datasets_from_files(search_dir, keys=keys, dtype=np.float64)
    concat_dataset : torchdata.ConcatDataset = torchdata.ConcatDataset(datasets)
    batch_size = trainerconfig["batch_size"]
    dataloader : torchdata.DataLoader = torchdata.DataLoader(concat_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=workers)
    Nfuture = datasets[0].metadata["numsamples_prediction"]
    tfuture = datasets[0].metadata["predictiontime"]
    tsegs : torch.Tensor = torch.linspace(0.0, tfuture, steps=num_segments+1, device=device, dtype=dtype)
    tstart_ = tsegs[:-1]
    dt_ = tsegs[1:] - tstart_
    tsamp_ : torch.Tensor = torch.linspace(0.0, tfuture, steps=Nfuture, device=device, dtype=dtype)
    coordinate_idx_history = [0,1]
    quaternion_idx_history = [2,3]
    for epoch in range(1, num_epochs+1):
        dataloader_enumerate = enumerate(dataloader)
        if comet_experiment is None:
            tq : tqdm.tqdm  = tqdm.tqdm(dataloader_enumerate, desc="Yay")
        else:
            tq : enumerate  = dataloader_enumerate
        print("Running epoch %d" % (epoch,), flush=True)
        if comet_experiment is not None:
            comet_experiment.set_epoch(epoch)    
        for (i, dict_) in tq:
            datadict : dict[str,torch.Tensor] = dict_
            position_history = datadict["hist"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
            vel_history = datadict["hist_vel"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)


            quat_history = datadict["hist_quats"][:,:,quaternion_idx_history].to(device=device, dtype=dtype)
            quat_history = quat_history/torch.norm(quat_history, p=2.0, dim=-1, keepdim=True)
            quat_history = quat_history*(torch.sign(quat_history[:,:,-1])[...,None])
            if heading_input_quaternion:
                quat_input = quat_history
            else:
                qz = quat_history[:,:,-2]
                qw = quat_history[:,:,-1]
                quat_input = 2.0*torch.atan2(qz,qw).unsqueeze(-1)


            position_future = datadict["fut"].to(device=device, dtype=dtype)
            vel_future = datadict["fut_vel"].to(device=device, dtype=dtype)
            left_bound_input = datadict["left_bd"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
            right_bound_input = datadict["right_bd"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
            left_bound_tangents = datadict["left_bd_tangents"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
            right_bound_tangents = datadict["right_bd_tangents"][:,:,coordinate_idx_history].to(device=device, dtype=dtype)
            left_bound_tangents = left_bound_tangents/torch.norm(left_bound_tangents, p=2.0, dim=-1, keepdim=True)
            right_bound_tangents = right_bound_tangents/torch.norm(right_bound_tangents, p=2.0, dim=-1, keepdim=True)

            history_inputs = torch.cat([position_history, vel_history, quat_input], dim=-1)
            left_boundary_inputs = torch.cat([left_bound_input, left_bound_tangents], dim=-1)
            right_boundary_inputs = torch.cat([right_bound_input, right_bound_tangents], dim=-1)
            
            p0 = position_future[:,0,coordinate_idx_history]
            v0 = vel_future[:,0,coordinate_idx_history]
            currentbatchdim = p0.shape[0]
            dt = dt_[None].expand(currentbatchdim, num_segments)
            velcurveout, poscurveout = network(history_inputs, left_boundary_inputs, right_boundary_inputs, dt, v0, p0=p0)
            tstart = tstart_[None].expand(currentbatchdim, num_segments)
            tsamp = tsamp_[None].expand(currentbatchdim, Nfuture)
            pout, _ = deepracing_models.math_utils.compositeBezierEval(tstart, dt, poscurveout, tsamp)
            deltas = pout - position_future[:,:,coordinate_idx_history]
            loss = torch.mean(torch.norm(deltas, p=2.0, dim=-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if comet_experiment is None:
                tq.set_postfix({"loss" : loss.item()})
            elif (i%20)==0:
                comet_experiment.log_metric("ade", loss.item())
        model_file = os.path.join(tempdir, "model_%d.pt" % (epoch,))
        optimizer_file = os.path.join(tempdir, "optimizer_%d.pt" % (epoch,))
        with open(model_file, "wb") as f:
            torch.save(network.state_dict(), f)
        with open(optimizer_file, "wb") as f:
            torch.save(optimizer.state_dict(), f)
        if comet_experiment is not None:
            comet_experiment.log_asset(model_file, file_name=os.path.basename(model_file), copy_to_tmp=False)  
            comet_experiment.log_asset(optimizer_file, file_name=os.path.basename(optimizer_file), copy_to_tmp=False)  

    



def prepare_and_train(argdict : dict):
    tempdir = argdict["tempdir"]
    workers = argdict["workers"]
    config_file = argdict["config_file"]
    gpu = argdict["gpu"]
    with open(config_file, "r") as f:
        config : dict = yaml.load(f, Loader=yaml.SafeLoader)
    api_key=os.getenv("COMET_API_KEY")
    if (api_key is not None) and len(api_key)>0:
        comet_experiment = comet_ml.Experiment(api_key=api_key, 
                                               project_name="bamf", 
                                               workspace="electric-turtle",
                                               auto_metric_logging=False
                                               )
        comet_experiment.log_asset(config_file, file_name="config.yaml", overwrite=True)
    else:
        comet_experiment = None
    train(config=config, workers=workers, tempdir=tempdir, comet_experiment = comet_experiment, gpu=gpu)

    
            
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train Bezier version of MixNet")
    parser.add_argument("config_file", type=str,  help="Configuration file to load")
    parser.add_argument("--tempdir", type=str, default=os.path.join(os.getenv("BIGTEMP", "/bigtemp/ttw2xk"), "bamf"), help="Temporary directory to save model files before uploading to comet. Default is to use tempfile module to generate one")
    parser.add_argument("--workers", type=int, default=0, help="How many threads for data loading")
    parser.add_argument("--gpu", type=int, default=-1, help="Which gpu???")
    args = parser.parse_args()
    argdict : dict = vars(args)
    prepare_and_train(argdict)