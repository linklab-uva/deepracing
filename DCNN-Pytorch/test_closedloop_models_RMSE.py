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
import deepracing_models.nn_models.LossFunctions as LF
import deepracing_models.data_loading.proto_datasets as PD
import deepracing_models.math_utils.bezier as bu
def main():
    parser = argparse.ArgumentParser(description="Test Closed Loop Models")
    parser.add_argument("waypoint_model_file", type=str)
    parser.add_argument("bezier_model_file", type=str)
    parser.add_argument("dataset_file", type=str)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--write_images", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    gpu = args.gpu

    #load waypoint predictor stuff
    waypoint_model_file = args.waypoint_model_file
    dataset_file = args.dataset_file
    waypoint_model_dir = os.path.dirname(waypoint_model_file)
    waypoint_config_file = os.path.join(waypoint_model_dir,'config.yaml')
    with open(waypoint_config_file,'r') as f:
        waypoint_config = yaml.load(f, Loader=yaml.SafeLoader)
    print(waypoint_config)
    waypoint = models.AdmiralNetKinematicPredictor(input_channels=waypoint_config["input_channels"], output_dimension=2, sequence_length=waypoint_config["sequence_length"], context_length=waypoint_config["context_length"], hidden_dim=waypoint_config["hidden_dimension"])
    with open(waypoint_model_file,'rb') as f:
        waypoint.load_state_dict(torch.load(f,map_location=torch.device("cpu")))
    waypoint = waypoint.double().cuda(gpu)

    
    #load bezier predictor stuff
    bezier_model_file = args.bezier_model_file
    dataset_file = args.dataset_file
    bezier_model_dir = os.path.dirname(bezier_model_file)
    bezier_config_file = os.path.join(bezier_model_dir,'config.yaml')
    with open(bezier_config_file,'r') as f:
        bezier_config = yaml.load(f, Loader=yaml.SafeLoader)
    print(bezier_config)
    if bezier_config["context_length"] == waypoint_config["context_length"]:
        print("Context length is the same for both models. All is well.")
    else:
        raise ValueError("Cannot test two models with different context lengths in the open-loop sense.")
    bezier = models.AdmiralNetCurvePredictor(input_channels=bezier_config["input_channels"], output_dimension=2, params_per_dimension=bezier_config["bezier_order"]+1, \
        context_length=bezier_config["context_length"], hidden_dim=bezier_config["hidden_dimension"])
    with open(bezier_model_file,'rb') as f:
        bezier.load_state_dict(torch.load(f,map_location=torch.device("cpu")))
    bezier = bezier.double().cuda(gpu)
    

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
        label_folder = os.path.join(dataset_root,"pose_sequence_labels")
        label_lmdb = os.path.join(label_folder,"lmdb")
        key_file = os.path.join(dataset_root,dataset["key_file"])

        label_wrapper = deepracing.backend.PoseSequenceLabelLMDBWrapper()
        label_wrapper.readDatabase(label_lmdb, mapsize=3e9, max_spare_txns=max_spare_txns )

        image_mapsize = float(np.prod(image_size)*3+12)*float(len(label_wrapper.getKeys()))*1.1
        image_wrapper = deepracing.backend.ImageLMDBWrapper(direct_caching=False)
        image_wrapper.readDatabase(image_lmdb, max_spare_txns=max_spare_txns, mapsize=image_mapsize )
        
        curent_dset = PD.PoseSequenceDataset(image_wrapper, label_wrapper, key_file, waypoint_config["context_length"], image_size = image_size, geometric_variants=False, )
        dsets.append(curent_dset)
    if len(dsets)==1:
        dset = dsets[0]
    else:
        dset = torch.utils.data.ConcatDataset(dsets)
    dataloader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False, pin_memory=True)
    numsamples = len(dataloader)
    print("Running test on %d image sequences" %(numsamples,))
    t = tqdm(dataloader)
    diffs_waypoint = []
    diffs_bezier =  []
    waypoint.eval()
    bezier.eval()
    s = torch.linspace(0,1,60).double().cuda(gpu).unsqueeze(0)
    Mbezier = bu.bezierM(s, bezier.params_per_dimension-1)
    loss = LF.SquaredLpNormLoss()

    for (i, (image_torch, opt_flow_torch, positions_torch, quats_torch, linear_velocities_torch, angular_velocities_torch, session_times_torch) ) in enumerate(t):

        image_torch_cuda = image_torch.double().cuda(gpu)
        image_torch_cuda.requires_grad_(False)

        positions_torch_cuda = positions_torch[:, : , [0,2] ].double().cuda(gpu)
        positions_torch_cuda.requires_grad_(False)

        numpoints = positions_torch.shape[1]
        if numpoints==waypoint.sequence_length:
            positions_torch_cuda_decimated = positions_torch_cuda
        else:
            pointsx = positions_torch_cuda[:,:,0]
            pointsz = positions_torch_cuda[:,:,1]
            pointsx_interp = torch.nn.functional.interpolate(pointsx.view(-1,1,numpoints), size=waypoint.sequence_length, scale_factor=None, mode='linear', align_corners=None).transpose(1,2).squeeze().unsqueeze(0)
            pointsz_interp = torch.nn.functional.interpolate(pointsz.view(-1,1,numpoints), size=waypoint.sequence_length, scale_factor=None, mode='linear', align_corners=None).transpose(1,2).squeeze().unsqueeze(0)
           # print(pointsx_interp.shape)
           # print(pointsz_interp.shape)
            positions_torch_cuda_decimated = torch.stack([pointsx_interp,pointsz_interp],dim=2)
        positions_torch_cuda_decimated.requires_grad_(False)
       # print(positions_torch_cuda_decimated.shape)

        predicted_control_points = bezier(image_torch_cuda).transpose(1,2)
       # print(predicted_control_points.shape)
        predictionsbezier = torch.matmul( Mbezier, predicted_control_points )
        #print(predictionsbezier.shape)
        predictionswaypoint = waypoint(image_torch_cuda)
        #print(predictionswaypoint.shape)
        loss_bezier = loss(predictionsbezier, positions_torch_cuda)
        loss_waypoint = loss(predictionswaypoint, positions_torch_cuda_decimated)
        diffs_waypoint.append(loss_waypoint.item())
        diffs_bezier.append(loss_bezier.item())





        #break
    diffs_waypoint_torch = torch.from_numpy(np.array(diffs_waypoint))
    print(diffs_waypoint_torch.shape)
    diffs_bezier_torch = torch.from_numpy(np.array(diffs_bezier))
    print(diffs_bezier_torch.shape)



    print("Waypoint:")
    print("RMSE diff: %f" %(torch.sqrt(torch.mean(diffs_waypoint_torch))))

    print("Bezier:")
    print("RMSE diff: %f" %(torch.sqrt(torch.mean(diffs_bezier_torch))))


if __name__ == '__main__':
    main()
