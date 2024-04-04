import os
import tempfile
bigtemp = os.getenv("BIGTEMP")
if bigtemp is not None:
    tempfiledir = os.path.join(bigtemp, "scratch")
    os.makedirs(tempfiledir, exist_ok=True)
    tempfile.tempdir=tempfiledir
import argparse
import comet_ml
from deepracing_models.data_loading import file_datasets as FD, SubsetFlag 
from deepracing_models.data_loading.utils.file_utils import load_datasets_from_files
from deepracing_models.nn_models.trajectory_prediction import BARTE
import torch, torch.nn, torch.utils.data as torchdata
import deepracing_models.training_utils as train_utils
import yaml
import numpy as np
import traceback
import sys
import deepracing_models.math_utils
import tqdm


def errorcb(exception):
    for elem in traceback.format_exception(exception):
        print(elem, flush=True, file=sys.stderr)

def train(config : dict = None, tempdir : str = None, num_epochs : int = 200, 
          workers : int = 0, gpu : int = -1, no_comet=False):

    if config is None:
        raise ValueError("keyword arg \"config\" is mandatory")
    if tempdir is None:
        raise ValueError("keyword arg \"tempdir\" is mandatory")
    
    keys : set = {
        "hist",
        "hist_quats",
        "hist_vel",
        "left_bd",
        "left_bd_tangents",
        "right_bd",
        "right_bd_tangents",
        "fut",
        "fut_vel",
        "fut_tangents"
    }
    trainer : train_utils.BarteTrainer = train_utils.BarteTrainer(config, tempdir, gpu=gpu, no_comet=no_comet)
    trainer.init_training_data(keys)
    trainer.init_validation_data(keys)
    trainer.build_optimizer()
    for epoch_number in range(1, trainer.trainerconfig["epochs"]+1):
        print("Running epoch %d" % (epoch_number,))
        trainer.run_epoch(epoch_number, train=True, workers=workers, with_tqdm=no_comet)
        trainer.checkpoint(epoch_number)
        trainer.run_epoch(epoch_number, train=False, workers=workers, with_tqdm=no_comet)

def prepare_and_train(argdict : dict):
    tempdir = argdict["tempdir"]
    workers = argdict["workers"]
    config_file = argdict["config_file"]
    gpu = argdict["gpu"]
    no_comet = argdict["no_comet"]
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    train(config=config, workers=workers, tempdir=tempdir, gpu=gpu, no_comet=no_comet)

    
            
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train Bezier version of MixNet")
    parser.add_argument("config_file", type=str,  help="Configuration file to load")
    parser.add_argument("--tempdir", type=str, default=os.path.join(os.getenv("BIGTEMP", "/bigtemp/ttw2xk"), "barte"), help="Temporary directory to save model files before uploading to comet. Default is to use tempfile module to generate one")
    parser.add_argument("--workers", type=int, default=0, help="How many threads for data loading")
    parser.add_argument("--gpu", type=int, default=-1, help="Which gpu???")
    parser.add_argument("--no-comet", action="store_true", help="Dont push to comet.ml")
    args = parser.parse_args()
    argdict : dict = vars(args)
    prepare_and_train(argdict)