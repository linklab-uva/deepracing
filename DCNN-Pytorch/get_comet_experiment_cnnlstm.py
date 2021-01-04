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
output_directory = os.path.join(args.output_directory, experiment_key)
if not os.path.isdir(output_directory):
    os.makedirs(output_directory)
if restkey is None:
    api = API()
else:
    api = API(rest_api_key=restkey)
experiment : APIExperiment = api.get_experiment("electric-turtle", "deepracingcnnlstm", experiment_key)
assetlist = experiment.get_asset_list()
assetdict = {d['fileName']: d['assetId'] for d in assetlist}
#print(assetdict)
#get network weight file
weightfilename = "CNNLSTM_epoch_%d_params.pt" %(epoch_number,)
weightfilenamealt = "epoch_%d_params.pt" %(epoch_number,)
weightfilenamealtalt = "pilotnet_epoch_%d_params.pt" %(epoch_number,)

print("Getting network weights from comet")
try:
    params_binary = experiment.get_asset(assetdict[weightfilename])
except KeyError as e:
    try:
        params_binary = experiment.get_asset(assetdict[weightfilename])
    except KeyError as e:
        params_binary = experiment.get_asset(assetdict[weightfilenamealtalt])
outputweightfile = os.path.join(output_directory,"epoch_%d_params.pt" %(epoch_number,))
    


with open(outputweightfile, 'wb') as f:
    f.write(params_binary)

#get parameters
parameters_summary = experiment.get_parameters_summary()
configin = {d["name"] : d["valueCurrent"] for d in parameters_summary}
config = {}
config["image_size"] = np.fromstring( configin["image_size"].replace(" ","")[1:-1], sep=',', dtype=np.int32).tolist()
config["input_channels"] = int( configin["input_channels"] )
config["output_dimension"] = int( configin.get("output_dimension","2") )
config["hidden_dimension"] = int( configin.get("hidden_dimension","100") )
config["sequence_length"] = int( configin["sequence_length"] )
config["context_length"] = int( configin.get("context_length","5") )
print(config)


outputconfigfile = os.path.join(output_directory,"config.yaml")
with open(outputconfigfile, 'w') as f:
    yaml.dump(config,stream=f,Dumper=yaml.SafeDumper)
input_channels = config["input_channels"]
context_length = config["context_length"]
sequence_length = config["sequence_length"]
output_dimension = config["output_dimension"]
hidden_dimension = config["hidden_dimension"]
net = deepracing_models.nn_models.Models.CNNLSTM(input_channels=input_channels,context_length=context_length, output_dimension=output_dimension, sequence_length=sequence_length, hidden_dimension=hidden_dimension) 
net = net.double()
with open(outputweightfile, 'rb') as f:
    net.load_state_dict(torch.load(f, map_location=torch.device("cpu")))
imtest = torch.rand(1,context_length,3,66,200).double()
output = net(imtest)
print(output)
print(output.shape)
