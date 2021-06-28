import argparse
import matplotlib.pyplot as plt
import cv2
import scipy
import numpy as np
import torch, torch.utils.data
import deepracing
import deepracing_models.data_loading.proto_datasets as PD
import os
import yaml
import deepracing_models.nn_models.StateEstimationModels as SEM
from tqdm import tqdm as tqdm

def main():
    parser = argparse.ArgumentParser(description="Test AdmiralNet")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--write_images", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    argdict = vars(args)

    root_folder = argdict["dataset"]
    dset = PD.PoseVelocityDataset(root_folder)
    dataloader = torch.utils.data.DataLoader(dset, batch_size = 1, shuffle=False)

    modelfile = argdict["model"]
    modeldir = os.path.dirname(modelfile)
    with open(os.path.join(modeldir, "model_config.yaml"), "r") as f:
        model_config = yaml.load(f, Loader=yaml.SafeLoader)
    with open(os.path.join(modeldir, "training_config.yaml"), "r") as f:
        training_config = yaml.load(f, Loader=yaml.SafeLoader)
        
    bezier_order = model_config["bezier_order"]
    bidirectional = model_config["bidirectional"]
    dropout = model_config["dropout"]
    hidden_dim = model_config["hidden_dim"]
    include_rotations = model_config.get("include_rotations", False)
    num_layers = model_config["num_layers"]
    input_dim = model_config["input_dim"]
    output_dim = model_config["output_dim"]
    net = SEM.ExternalAgentCurvePredictor(output_dim=output_dim, bezier_order=bezier_order, input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
    net = net.float()
    with open(modelfile, 'rb') as f:
        net.load_state_dict(torch.load(f, map_location=torch.device("cpu")))
    net = net.cuda(0)
    t : tqdm = tqdm(dataloader)
    dev = next(net.parameters()).device
    dtype = next(net.parameters()).dtype 
    for (i, datadict) in enumerate(t):
        valid_mask = datadict["valid_mask"] 
        past_positions = datadict["past_positions"]
        past_velocities = datadict["past_velocities"]
        future_positions = datadict["future_positions"]
        tfuture = datadict["tfuture"]

        valid_past_positions = (past_positions[valid_mask].type(dtype).to(dev))[:,:,[0,2]]
        valid_past_velocities = (past_velocities[valid_mask].type(dtype).to(dev))[:,:,[0,2]]
        valid_future_positions = (future_positions[valid_mask].type(dtype).to(dev))[:,:,[0,2]]
        valid_tfuture = tfuture[valid_mask].type(dtype).to(dev)

        # print(valid_past_positions.shape)
        # print(valid_future_positions.shape)
        past_pos_plot = valid_past_positions[0].cpu().numpy()
        future_pos_plot = valid_future_positions[0].cpu().numpy()
        figure = plt.figure()
        plt.xlabel("X (meters)")
        plt.ylabel("Z (meters)")
        plt.plot(past_pos_plot[:,0], past_pos_plot[:,1], label="Past Positions", c="b")
        plt.plot(future_pos_plot[:,0], future_pos_plot[:,1], label="Future Positions", c="g")
        plt.legend()
        plt.show()

        

    



if __name__=="__main__":
    main()
    