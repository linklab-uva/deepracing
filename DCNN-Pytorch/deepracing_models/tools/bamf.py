import argparse
# import comet_ml
# import deepracing_models
import deepracing_models.math_utils.bezier, deepracing_models.math_utils
from deepracing_models.nn_models.trajectory_prediction import BAMF
# from deepracing_models.data_loading import file_datasets as FD, SubsetFlag 
# from deepracing_models.data_loading.utils.file_utils import load_datasets_from_files, load_datasets_from_shared_memory
import torch, torch.nn
import yaml
import os
import numpy as np
import traceback
import sys


def errorcb(exception):
    for elem in traceback.format_exception(exception):
        print(elem, flush=True, file=sys.stderr)

def train(config : dict = None, tempdir : str = None, num_epochs : int = 200, 
          workers : int = 0, api_key=None, dtype : np.dtype | None = None):

    if tempdir is None:
        raise ValueError("keyword arg \"tempdir\" is mandatory")
    

    network : BAMF = BAMF().double()
    lossfunc : torch.nn.MSELoss = torch.nn.MSELoss().double()

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