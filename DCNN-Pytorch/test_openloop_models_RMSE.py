import cv2
import glob
import numpy as np
import deepracing_models.math_utils
import deepracing_models.nn_models.Models as models
import numpy.random
import torch, random
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm as tqdm
import yaml
from datetime import datetime
import os
import string
import argparse
from random import randint
from datetime import datetime
import torchvision.transforms as transforms
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import deepracing, deepracing_models
import deepracing.backend.LabelBackends
import deepracing_models.data_loading.proto_datasets as PD
def main():
    parser = argparse.ArgumentParser(description="Test Open Loop Models")
    parser.add_argument("pilotnet_model_file", type=str)
    parser.add_argument("cnnlstm_model_file", type=str)
    parser.add_argument("admiralnet_e2e_model_file", type=str)
    parser.add_argument("dataset_file", type=str)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--write_images", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    gpu = args.gpu

    #load pilotnet stuff
    pilotnet_model_file = args.pilotnet_model_file
    dataset_file = args.dataset_file
    pilotnet_model_dir = os.path.dirname(pilotnet_model_file)
    pilotnet_config_file = os.path.join(pilotnet_model_dir,'config.yaml')
    with open(pilotnet_config_file,'r') as f:
        pilotnet_config = yaml.load(f, Loader=yaml.SafeLoader)
    print(pilotnet_config)
    pilotnet = models.PilotNet(input_channels=pilotnet_config["input_channels"], output_dim=pilotnet_config["output_dimension"])
    with open(pilotnet_model_file,'rb') as f:
        pilotnet.load_state_dict(torch.load(f,map_location=torch.device("cpu")))
    pilotnet = pilotnet.double().cuda(gpu)

    
    #load cnnlstm stuff
    cnnlstm_model_file = args.cnnlstm_model_file
    dataset_file = args.dataset_file
    cnnlstm_model_dir = os.path.dirname(cnnlstm_model_file)
    cnnlstm_config_file = os.path.join(cnnlstm_model_dir,'config.yaml')
    with open(cnnlstm_config_file,'r') as f:
        cnnlstm_config = yaml.load(f, Loader=yaml.SafeLoader)
    print(cnnlstm_config)
    cnnlstm = models.CNNLSTM(input_channels=cnnlstm_config["input_channels"], output_dimension=cnnlstm_config["output_dimension"],context_length=cnnlstm_config["context_length"],sequence_length=cnnlstm_config["sequence_length"], hidden_dimension=cnnlstm_config["hidden_dimension"])
    with open(cnnlstm_model_file,'rb') as f:
        cnnlstm.load_state_dict(torch.load(f,map_location=torch.device("cpu")))
    cnnlstm = cnnlstm.double().cuda(gpu)

    #load end-to-end AdmiralNet stuff
    admiralnet_e2e_model_file = args.admiralnet_e2e_model_file
    dataset_file = args.dataset_file
    admiralnet_e2e_model_dir = os.path.dirname(admiralnet_e2e_model_file)
    admiralnet_e2e_config_file = os.path.join(admiralnet_e2e_model_dir,'config.yaml')
    with open(admiralnet_e2e_config_file,'r') as f:
        admiralnet_e2e_config = yaml.load(f, Loader=yaml.SafeLoader)
    print(admiralnet_e2e_config)
    admiralnet_e2e = models.AdmiralNetKinematicPredictor(input_channels=admiralnet_e2e_config["input_channels"], output_dimension=admiralnet_e2e_config["output_dimension"],context_length=admiralnet_e2e_config["context_length"],sequence_length=admiralnet_e2e_config["sequence_length"], hidden_dim=admiralnet_e2e_config["hidden_dimension"])
    with open(admiralnet_e2e_model_file,'rb') as f:
        admiralnet_e2e.load_state_dict(torch.load(f,map_location=torch.device("cpu")))
    admiralnet_e2e = admiralnet_e2e.double().cuda(gpu)
    

    with open(dataset_file,'r') as f:
        dataset_config = yaml.load(f, Loader=yaml.SafeLoader)
    image_size = np.array(dataset_config["image_size"])
    datasets = dataset_config["datasets"]
    dsets=[]
    max_spare_txns = 50
    for dataset in datasets:
        print("Parsing database config: %s" %(str(dataset)))
        dataset_root = dataset["dataset_root"]
        image_folder = os.path.join(dataset_root,"images")
        image_lmdb = os.path.join(image_folder,"image_lmdb")
        optflow_lmdb = os.path.join(image_folder,"optical_flow_lmdb")
        label_folder = os.path.join(dataset_root,"steering_labels")
        label_lmdb = os.path.join(label_folder,"lmdb")
        key_file = os.path.join(dataset_root,dataset["key_file"])

        label_wrapper = deepracing.backend.ControlLabelLMDBWrapper()
        label_wrapper.readDatabase(label_lmdb, mapsize=3e9, max_spare_txns=max_spare_txns )

        image_mapsize = float(np.prod(image_size)*3+12)*float(len(label_wrapper.getKeys()))*1.1
        image_wrapper = deepracing.backend.ImageLMDBWrapper(direct_caching=False)
        image_wrapper.readDatabase(image_lmdb, max_spare_txns=max_spare_txns, mapsize=image_mapsize )

        optflow_wrapper = deepracing.backend.OpticalFlowLMDBWrapper()
        optflow_wrapper.readDatabase( optflow_lmdb, max_spare_txns=max_spare_txns, mapsize=4*image_mapsize )
        
        curent_dset = PD.ControlOutputSequenceDataset(image_wrapper, label_wrapper, key_file, image_size = image_size, optflow_db_wrapper=optflow_wrapper)
        dsets.append(curent_dset)
    if len(dsets)==1:
        dset = dsets[0]
    else:
        dset = torch.utils.data.ConcatDataset(dsets)
    dataloader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False, pin_memory=True)
    numsamples = len(dataloader)
    print("Running test on %d images" %(numsamples,))
    t = tqdm(dataloader)
    diffs_pilot = []
    diffs_cnnlstm =  []
    diffs_ame2e =  []
    pilotnet.eval()
    cnnlstm.eval()
    admiralnet_e2e.eval()
    for (i, (image_torch, optflow_torch, control_output) ) in enumerate(t):
        image_torch_cuda = image_torch.double().cuda(gpu)
        image_torch_cuda.requires_grad_(False)
        optflow_torch_cuda = optflow_torch.double().cuda(gpu)
        optflow_torch_cuda.requires_grad_(False)
        control_output_cuda = control_output[:,0,:].squeeze().double().cuda(gpu)
        control_output_cuda.requires_grad_(False)

        predictionscnnlstm = cnnlstm(image_torch_cuda)[:,0,:].squeeze()
        diffs_cnnlstm.append((predictionscnnlstm - control_output_cuda).detach().cpu().numpy())

        
        predictionsame2e = admiralnet_e2e( torch.cat( (image_torch_cuda, optflow_torch_cuda), dim=2 ) )[:,0,:].squeeze()
        diffs_ame2e.append((predictionsame2e - control_output_cuda).detach().cpu().numpy())
    # for (i, (image_torch, control_output) ) in enumerate(t):
    #     image_torch_cuda = image_torch.double().cuda(gpu)
    #     control_output_cuda = control_output[:,0,:].squeeze().double().cuda(gpu)

        predictionspilotnet = pilotnet(image_torch_cuda[:,-1,:,:,:]).squeeze()
        diffs_pilot.append((predictionspilotnet - control_output_cuda).detach().cpu().numpy())
        #break
    diffs_pilot_torch = torch.from_numpy(np.array(diffs_pilot))
    print(diffs_pilot_torch.shape)
    diffs_cnnlstm_torch = torch.from_numpy(np.array(diffs_cnnlstm))
    print(diffs_cnnlstm_torch.shape)
    diffs_ame2e_torch = torch.from_numpy(np.array(diffs_ame2e))
    print(diffs_ame2e_torch.shape)



    print("PilotNet:")
    print("RMSE steering: %f" %(torch.sqrt(torch.mean(torch.pow(diffs_pilot_torch[:,0],2)))))
    print("RMSE throttle: %f" %(torch.sqrt(torch.mean(torch.pow(diffs_pilot_torch[:,1],2)))))

    print("CNNLSTM:")
    print("RMSE steering: %f" %(torch.sqrt(torch.mean(torch.pow(diffs_cnnlstm_torch[:,0],2)))))
    print("RMSE throttle: %f" %(torch.sqrt(torch.mean(torch.pow(diffs_cnnlstm_torch[:,1],2)))))

    print("End-to-end AdmiralNet:")
    print("RMSE steering: %f" %(torch.sqrt(torch.mean(torch.pow(diffs_ame2e_torch[:,0],2)))))
    print("RMSE throttle: %f" %(torch.sqrt(torch.mean(torch.pow(diffs_ame2e_torch[:,1],2)))))

if __name__ == '__main__':
    main()
