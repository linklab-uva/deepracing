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
import deepracing_models.nn_models.Models, deepracing_models.nn_models.VariationalModels
import io

parser = argparse.ArgumentParser(description="Download experiment from Comet.ml")
parser.add_argument("experiment_key", type=str, help="Experiment key to grab from comet.")
parser.add_argument("--api_key", type=str, required=False, default=None, help="Experiment key to grab from comet.")
parser.add_argument("--epoch_number", type=int, required=False, default=200, help="Experiment key to grab from comet.")
parser.add_argument("--output_directory", type=str, required=False, default=os.curdir, help="Where to put the config and data files.")
parser.add_argument("--get_optimizer_weights", action="store_true", help="Also grab the state dictionary of the optimizer at the end of the specified epoch")

#XMUs9uI19KQNdYrQhuXPnAfpB
args = parser.parse_args() 
experiment_key = args.experiment_key
epoch_number = args.epoch_number
apikey = args.api_key
get_optimizer_weights = args.get_optimizer_weights
output_directory = os.path.join(args.output_directory, experiment_key )
if not os.path.isdir(output_directory):
    os.makedirs(output_directory)
api = API(api_key=apikey)
    
experiment : APIExperiment = api.get_experiment("electric-turtle", "probabilisticbeziercurves", experiment_key)
assetlist = experiment.get_asset_list()
assetdict = {d['fileName']: d['assetId'] for d in assetlist}
#lookahead_indices = int(experiment.get_parameters_summary(parameter="lookahead_indices")["valueCurrent"])


print("Getting hyperparameters from comet")
datasets_file_name = "datasets.yaml"
experiment_file_name = "experiment_config.yaml"
config_file_name = "model_config.yaml"
config_yaml = experiment.get_asset(assetdict[config_file_name], return_type="text")
config = yaml.load(io.StringIO(config_yaml), Loader=yaml.SafeLoader)
print("Got config from comet")
print(config)
datasets_yaml = experiment.get_asset(assetdict[datasets_file_name], return_type="text")
datasets_dict = yaml.load(io.StringIO(datasets_yaml), Loader=yaml.SafeLoader)
outputconfigfile = os.path.join(output_directory,config_file_name)
outputexperimentfile = os.path.join(output_directory,experiment_file_name)
outputdatasetsfile = os.path.join(output_directory,datasets_file_name)
with open(outputdatasetsfile, 'w') as f:
    yaml.dump(datasets_dict, f, Dumper=yaml.SafeDumper)
with open(outputconfigfile, 'w') as f:
    yaml.dump(config, f, Dumper=yaml.SafeDumper)
with open(outputexperimentfile, 'w') as f:
    yaml.dump({"experiment_key" : experiment_key})#, "lookahead_indices": lookahead_indices}, f, Dumper=yaml.SafeDumper)


#get network weight file
weightfilename = "epoch_%d_params.pt" %(epoch_number,)

print("Getting network weights from comet")
params_binary = experiment.get_asset(assetdict[weightfilename])

outputweightfile = os.path.join(output_directory,weightfilename)
with open(outputweightfile, 'wb') as f:
    f.write(params_binary)

#get optimizer weight file
if get_optimizer_weights:
    print("Getting optimizer weights from comet")
    optimizerfilename = "epoch_%d_optimizer.pt" %(epoch_number,)
    optimizer_binary = experiment.get_asset(assetdict[optimizerfilename])
    outputoptimizerfile = os.path.join(output_directory,optimizerfilename)
    with open(outputoptimizerfile, 'wb') as f:
        f.write(optimizer_binary)

print("Attempting to reconstruct model with configuration from comet")
context_length = config["context_length"]
input_channels = config["input_channels"]
hidden_dimension = config["hidden_dimension"]
bezier_order = config["bezier_order"]
num_recurrent_layers = config.get("num_recurrent_layers",1)
fix_first_point = config.get("fix_first_point",False)
net = deepracing_models.nn_models.VariationalModels.VariationalCurvePredictor( context_length = context_length , input_channels=input_channels, hidden_dim = hidden_dimension, num_recurrent_layers=num_recurrent_layers, fix_first_point=fix_first_point, bezier_order=bezier_order) 
with open(outputweightfile, 'rb') as f:
    net.load_state_dict(torch.load(f, map_location=torch.device("cpu")))