import comet_ml
import torch
import os
import argparse
from comet_ml.api import API, APIExperiment
import yaml
import torch.nn as NN
import numpy as np
import torch.utils.data as data_utils
from tqdm import tqdm as tqdm
import deepracing_models.nn_models.StateEstimationModels as SEM
import io

parser = argparse.ArgumentParser(description="Download experiment from Comet.ml")
parser.add_argument("experiment_key", type=str, help="Experiment key to grab from comet.")
parser.add_argument("--restkey", type=str, required=False, default=None, help="Experiment key to grab from comet.")
parser.add_argument("--epoch_number", type=int, required=False, default=100, help="Experiment key to grab from comet.")
parser.add_argument("--output_directory", type=str, required=False, default=os.curdir, help="Where to put the config and data files.")
parser.add_argument("--get_optimizer_weights", action="store_true", help="Also grab the state dictionary of the optimizer at the end of the specified epoch")

#XMUs9uI19KQNdYrQhuXPnAfpB
args = parser.parse_args() 
experiment_key = args.experiment_key
epoch_number = args.epoch_number
restkey = args.restkey
get_optimizer_weights = args.get_optimizer_weights
output_directory = os.path.join(args.output_directory, experiment_key )
if not os.path.isdir(output_directory):
    os.makedirs(output_directory)
if restkey is None:
    api = API()
else:
    api = API(rest_api_key=restkey)
    
experiment : APIExperiment = api.get_experiment("electric-turtle", "deepracingbezierpredictor", experiment_key)
assetdict = { d['fileName'] : d for d in experiment.get_asset_list() }

print("Getting hyperparameters from comet")
model_config_name = "model_config.yaml"
training_config_name = "training_config.yaml"
experiment_config_name = "experiment_config.yaml"
model_config_yaml = experiment.get_asset((assetdict[model_config_name])['assetId'], return_type="text")
model_config = yaml.load(io.StringIO(model_config_yaml), Loader=yaml.SafeLoader)
print("Got model config from comet")
print(model_config)
with open(os.path.join(output_directory, "model_config.yaml"), "w") as f:
    yaml.dump(model_config, f, Dumper=yaml.SafeDumper)
training_config_yaml = experiment.get_asset((assetdict[training_config_name])['assetId'], return_type="text")
training_config = yaml.load(io.StringIO(training_config_yaml), Loader=yaml.SafeLoader)
print("Got training config from comet")
print(training_config)
with open(os.path.join(output_directory, "training_config.yaml"), "w") as f:
    yaml.dump(training_config, f, Dumper=yaml.SafeDumper)



weightfilename = "epoch_%d_params.pt" %(epoch_number,)

print("Getting network weights from comet")
params_binary = experiment.get_asset((assetdict[weightfilename])['assetId'])

outputweightfile = os.path.join(output_directory,weightfilename)
with open(outputweightfile, 'wb') as f:
    f.write(params_binary)

print("Attempting to reconstruct model with configuration from comet")
bezier_order = model_config["bezier_order"]
bidirectional = model_config["bidirectional"]
dropout = model_config["dropout"]
hidden_dim = model_config["hidden_dim"]
include_rotations = model_config["include_rotations"]
num_layers = model_config["num_layers"]
input_dim = model_config["input_dim"]
output_dim = model_config["output_dim"]

net = SEM.ExternalAgentCurvePredictor(output_dim=output_dim, bezier_order=bezier_order, input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
print(net.state_dict())
print()
print()
with open(outputweightfile, 'rb') as f:
    net.load_state_dict(torch.load(f, map_location=torch.device("cpu")))
print(net.state_dict())