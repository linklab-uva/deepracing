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
output_directory = os.path.join(args.output_directory, experiment_key + "_from_comet")
if not os.path.isdir(output_directory):
    os.makedirs(output_directory)
if restkey is None:
    api = API()
else:
    api = API(rest_api_key=restkey)
    
experiment : APIExperiment = api.get_experiment("electric-turtle", "deepracingbezierpredictor", experiment_key)
assetlist = experiment.get_asset_list()
assetdict = {d['fileName']: d['assetId'] for d in assetlist}


print("Getting hyperparameters from comet")
parameters_summary = experiment.get_parameters_summary()
configin = {d["name"] : d["valueCurrent"] for d in parameters_summary}
config = {}
config["image_size"] = np.fromstring( configin["image_size"].replace(" ","").replace("[","").replace("]",""), sep=',', dtype=np.int32 ).tolist()
# try:
#     config["image_size"] = np.fromstring( configin["image_size"].replace(" ","").replace("[","").replace("]",""), sep=',', dtype=np.int32 ).tolist()
# except:
#     config["image_size"] = [66,200]
config["input_channels"] = int( configin["input_channels"] )
config["hidden_dimension"] = int( configin["hidden_dimension"] )
config["sequence_length"] = int( configin["sequence_length"] )
config["context_length"] = int( configin["context_length"] )
config["bezier_order"] = int( configin["bezier_order"] )
print(config)
outputconfigfile = os.path.join(output_directory,"config.yaml")
with open(outputconfigfile, 'w') as f:
    yaml.dump(config,stream=f,Dumper=yaml.SafeDumper)


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


context_length = int(config["context_length"])
input_channels = int(config["input_channels"])
hidden_dimension = int(config["hidden_dimension"])
bezier_order = int(config["bezier_order"])
num_recurrent_layers = int(config.get("num_recurrent_layers","1"))
net = deepracing_models.nn_models.Models.AdmiralNetCurvePredictor( context_length = context_length , input_channels=input_channels, hidden_dim = hidden_dimension, num_recurrent_layers=num_recurrent_layers, params_per_dimension=bezier_order+1 ) 
net = net.double()
with open(outputweightfile, 'rb') as f:
    net.load_state_dict(torch.load(f, map_location=torch.device("cpu")))