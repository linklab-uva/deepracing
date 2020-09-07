import comet_ml
import torch
import os
import argparse
from comet_ml.api import API, APIExperiment
import yaml
import torch.nn as NN
import numpy as np
import torch.utils.data as data_utils
import deepracing_models.data_loading.proto_datasets as PD
from tqdm import tqdm as tqdm
import deepracing_models.nn_models.LossFunctions as loss_functions
import deepracing_models.nn_models.Models
parser = argparse.ArgumentParser(description="Download experiment from Comet.ml")
parser.add_argument("experiment_key", type=str, help="Experiment key to grab from comet.")
parser.add_argument("--restkey", type=str, required=False, default=None, help="Experiment key to grab from comet.")
parser.add_argument("--epoch_number", type=int, required=False, default=100, help="Experiment key to grab from comet.")
parser.add_argument("--output_directory", type=str, required=False, default=".", help="Where to put the config and data files.")

#XMUs9uI19KQNdYrQhuXPnAfpB
args = parser.parse_args() 
experiment_key = args.experiment_key
epoch_number = args.epoch_number
restkey = args.restkey
output_directory = os.path.join(args.output_directory, experiment_key )
if not os.path.isdir(output_directory):
    os.makedirs(output_directory)
if restkey is None:
    api = API()
else:
    api = API(rest_api_key=restkey)
    
experiment : APIExperiment = api.get_experiment("electric-turtle", "deepracingbezierpredictor", experiment_key)
assetlist = experiment.get_asset_list()
assetdict = {d['fileName']: d['assetId'] for d in assetlist}
lookahead_indices = int(experiment.get_parameters_summary(parameter="lookahead_indices")["valueCurrent"])


print("Getting hyperparameters from comet")
experiment_file_name = "experiment_config.yaml"
config_file_name = "model_config.yaml"
config_yaml = experiment.get_asset(assetdict[config_file_name], return_type="text")
outputconfigfile = os.path.join(output_directory,config_file_name)
outputexperimentfile = os.path.join(output_directory,experiment_file_name)
with open(outputconfigfile, 'w') as f:
    f.write(config_yaml)
with open(outputexperimentfile, 'w') as f:
    yaml.dump({"experiment_key" : experiment_key, "lookahead_indices": lookahead_indices}, f, Dumper=yaml.SafeDumper)
with open(outputconfigfile, 'r') as f:
    config = yaml.load(f,Loader=yaml.SafeLoader)
print("Got config from comet")
print(config)


#get network weight file
weightfilename = "epoch_%d_params.pt" %(epoch_number,)
optimizerfilename = "epoch_%d_optimizer.pt" %(epoch_number,)

print("Getting network weights from comet")
params_binary = experiment.get_asset(assetdict[weightfilename])

outputweightfile = os.path.join(output_directory,weightfilename)
with open(outputweightfile, 'wb') as f:
    f.write(params_binary)

#get optimizer weight file
print("Getting optimizer weights from comet")
optimizer_binary = experiment.get_asset(assetdict[optimizerfilename])


outputoptimizerfile = os.path.join(output_directory,optimizerfilename)
with open(outputoptimizerfile, 'wb') as f:
    f.write(optimizer_binary)


context_length = config["context_length"]
input_channels = config["input_channels"]
hidden_dimension = config["hidden_dimension"]
bezier_order = config["bezier_order"]
num_recurrent_layers = config.get("num_recurrent_layers",1)
fix_first_point = config.get("fix_first_point",False)
net = deepracing_models.nn_models.Models.AdmiralNetCurvePredictor( context_length = context_length , input_channels=input_channels, hidden_dim = hidden_dimension, num_recurrent_layers=num_recurrent_layers, params_per_dimension=bezier_order+1-int(fix_first_point) ) 
net = net.double()
with open(outputweightfile, 'rb') as f:
    net.load_state_dict(torch.load(f, map_location=torch.device("cpu")))