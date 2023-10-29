import argparse
# import comet_ml
# import deepracing_models
# from deepracing_models.data_loading import file_datasets as FD, SubsetFlag 
# from deepracing_models.data_loading.utils.file_utils import load_datasets_from_files, load_datasets_from_shared_memory
import deepracing_models.math_utils
from deepracing_models.nn_models.trajectory_prediction import BAMF
import torch, torch.nn
import yaml
import os
import numpy as np
import traceback
import sys
import time


def errorcb(exception):
    for elem in traceback.format_exception(exception):
        print(elem, flush=True, file=sys.stderr)

def train(config : dict = None, tempdir : str = None, num_epochs : int = 200, 
          workers : int = 0, api_key=None, dtype : np.dtype | None = None):

    if tempdir is None:
        raise ValueError("keyword arg \"tempdir\" is mandatory")
    
    gpu = 3
    num_segments = 8
    kbezier = 3
    network : BAMF = BAMF(
            num_segments = num_segments, 
            kbezier = kbezier
        ).float().train().cuda(gpu)
    firstparam = next(network.parameters())
    device = firstparam.device
    dtype = firstparam.dtype
    lossfunc : torch.nn.MSELoss = torch.nn.MSELoss().to(device=device, dtype=dtype)
    batchdim=256
    t = torch.linspace(0.0, 3.0, steps=num_segments+1).to(device=device, dtype=dtype).unsqueeze(0).expand(batchdim, num_segments+1)
    nsamp = 61
    tsamp = torch.linspace(0.0, 3.0, steps=nsamp).to(device=device, dtype=dtype).unsqueeze(0).expand(batchdim, nsamp).clone()
    tstart = t[:,:-1]
    dt = t[:,1:] - tstart
    random_history = torch.randn(batchdim, 31, 4).to(device=device, dtype=dtype)
    p0 = random_history[:,-1,:2]
    v0 = random_history[:,-1,2:]
    random_lb = torch.randn(batchdim, nsamp, 4).to(device=device, dtype=dtype)
    random_rb = torch.randn(batchdim, nsamp, 4).to(device=device, dtype=dtype)
    random_gt = torch.randn(batchdim, nsamp, 2).to(device=device, dtype=dtype)
    print("here we go")
    tick = time.time()
    velcurveout, poscurveout = network(random_history, random_lb, random_rb, dt, p0, v0)
    pout, idxbuckets = deepracing_models.math_utils.compositeBezierEval(tstart, dt, poscurveout, tsamp)
    fakedeltas = pout - random_gt
    fakeloss = torch.mean(torch.norm(fakedeltas, dim=-1))
    fakeloss.backward()
    tock = time.time()
    print("done. took %f seconds" % (tock-tick,))
    print(tsamp.shape)
    print(poscurveout.shape)

    dataconfig = config["data"]

    # search_dirs = dataconfig["dirs"]
    # datasets = []
    # for search_dir in search_dirs:
    #     datasets += load_datasets_from_files(search_dir, dtype=dtype)
    # concat_dataset : torchdata.ConcatDataset = torchdata.ConcatDataset(datasets)
    # dataloader : torchdata.DataLoader = torchdata.DataLoader(concat_dataset, batch_size=64, pin_memory=True, shuffle=True, num_workers=workers)


def prepare_and_train(argdict : dict):
    tempdir = argdict["tempdir"]
    workers = argdict["workers"]
    config_file = argdict["config_file"]
    with open(config_file, "r") as f:
        config : dict = yaml.load(f, Loader=yaml.SafeLoader)
    train(config=config, workers=workers, tempdir=tempdir, api_key=os.getenv("COMET_API_KEY"))

    
            
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train Bezier version of MixNet")
    parser.add_argument("config_file", type=str,  help="Configuration file to load")
    parser.add_argument("--tempdir", type=str, required=True, help="Temporary directory to save model files before uploading to comet. Default is to use tempfile module to generate one")
    parser.add_argument("--workers", type=int, default=0, help="How many threads for data loading")
    args = parser.parse_args()
    argdict : dict = vars(args)
    prepare_and_train(argdict)