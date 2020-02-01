import comet_ml
import torch
import os
import argparse
from comet_ml.api import API, APIExperiment
import yaml
import torch.nn as NN
import torch.utils.data as data_utils
import deepracing_models.data_loading.proto_datasets as PD
from tqdm import tqdm as tqdm
import deepracing_models.nn_models.LossFunctions as loss_functions
import deepracing_models.nn_models.Models
parser = argparse.ArgumentParser(description="Download experiment from Comet.ml")
parser.add_argument("project_name", type=str, help="Project name to pull from.")
parser.add_argument("experiment_key", type=str, help="Experiment key to grab from comet.")
parser.add_argument("--restkey", type=str, required=False, default=None, help="Experiment key to grab from comet.")
parser.add_argument("--epoch_number", type=int, required=False, default=100, help="Experiment key to grab from comet.")
parser.add_argument("--output_directory", type=str, required=False, default=".", help="Where to put the config and data files.")

projectdict = {"cnnlstm" : "deepracingcnnlstm", "bezier" : "deepracingbezierpredictor", "pilot" : "deepracingpilotnet"}
#XMUs9uI19KQNdYrQhuXPnAfpB
args = parser.parse_args()
project_name = args.project_name
experiment_key = args.experiment_key
epoch_number = args.epoch_number
restkey = args.restkey
output_directory = args.output_directory
if restkey is None:
    api = API()
else:
    api = API(rest_api_key=restkey)
experiment : APIExperiment = api.get_experiment("electric-turtle", projectdict[project_name], experiment_key)
assetlist = experiment.get_asset_list()
assetdict = {d['fileName']: d['assetId'] for d in assetlist}
print(assetdict)
print("Getting weight file from COMET")
filename = "epoch_%d_params.pt" %(epoch_number,)
assetid = assetdict[filename]
params_binary = experiment.get_asset(assetid)
outputfile = os.path.join(output_directory,filename)
with open(outputfile, 'wb') as f:
    f.write(params_binary)
parameters_summary = experiment.get_parameters_summary()
config = {d["name"] : d["valueCurrent"] for d in parameters_summary}
print(config)
outputconfigfile = os.path.join(output_directory,"config.yaml")
with open(outputconfigfile, 'w') as f:
    yaml.dump(config,stream=f,Dumper=yaml.SafeDumper)

# context_length = int(config["context_length"])
# input_channels = int(config["input_channels"])
# hidden_dimension = int(config["hidden_dimension"])
# bezier_order = int(config["bezier_order"])
# num_recurrent_layers = int(config.get("num_recurrent_layers","1"))
# net = deepracing_models.nn_models.Models.AdmiralNetCurvePredictor( context_length = context_length , input_channels=input_channels, hidden_dim = hidden_dimension, num_recurrent_layers=num_recurrent_layers, params_per_dimension=bezier_order+1 ) 
# net = net.double()
# with open(outputfile, 'rb') as f:
#     net.load_state_dict(torch.load(f, map_location=torch.device("cpu")))