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
def assetkey(asset : dict):
    return asset["step"]
def test(**kwargs):
    experiment : str = kwargs["experiment"]
    tempdir : str = kwargs["tempdir"]
    workers : int = kwargs["workers"]
    
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
    print(os.listdir(experiment_dir))
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)
    net_file = os.path.join(experiment_dir, "net.pt")
    # if not os.path.isfile(net_file):



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Test Bezier version of MixNet")
    parser.add_argument("--experiment", type=str, required=True, help="Which comet experiment to load")
    parser.add_argument("--tempdir", type=str, default="/bigtemp/ttw2xk/mixnet_bezier_dump", help="Temporary directory to save model files after downloading from comet.")
    parser.add_argument("--workers", type=int, default=0, help="How many threads for data loading")
    args = parser.parse_args()
    argdict : dict = vars(args)
    test(**argdict)