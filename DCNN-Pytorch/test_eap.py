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
import deepracing_models.math_utils as mu

def main():
    parser = argparse.ArgumentParser(description="Test AdmiralNet")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--write_images", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    argdict = vars(args)


    modelfile = argdict["model"]
    modeldir = os.path.dirname(modelfile)
    with open(os.path.join(modeldir, "model_config.yaml"), "r") as f:
        model_config = yaml.load(f, Loader=yaml.SafeLoader)
    with open(os.path.join(modeldir, "training_config.yaml"), "r") as f:
        training_config = yaml.load(f, Loader=yaml.SafeLoader)

    context_indices = training_config.get("context_indices", 5)
    context_time = training_config.get("context_time", 1.75)
    prediction_time = training_config.get("prediction_time", 1.75)
    root_folder = argdict["dataset"]
    dset = PD.PoseVelocityDataset(root_folder, context_indices=context_indices, context_time=context_time, prediction_time=prediction_time)
    dataloader = torch.utils.data.DataLoader(dset, batch_size = 1, shuffle=True)
        
    bezier_order = model_config["bezier_order"]
    bidirectional = model_config["bidirectional"]
    dropout = model_config["dropout"]
    hidden_dim = model_config["hidden_dim"]
    include_rotations = model_config.get("include_rotations", False)
    num_layers = model_config["num_layers"]
    input_dim = model_config["input_dim"]
    output_dim = model_config["output_dim"]
    learnable_initial_state = model_config.get("learnable_initial_state",False)
    #network = SEM.ExternalAgentCurvePredictor(learnable_initial_state = learnable_initial_state, output_dim=output_dim, bezier_order=bezier_order, input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
    network = SEM.ProbabilisticExternalAgentCurvePredictor(learnable_initial_state = learnable_initial_state, output_dim=output_dim, bezier_order=bezier_order, input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
    
    network = network.float()
    with open(modelfile, 'rb') as f:
        # network.load_state_dict(torch.load(f, map_location=torch.device("cpu")))
        network.load_state_dict(torch.load(f, map_location=torch.device("cpu")), strict=False)
    network = network.cuda(0)
    print(network.state_dict())
    t : tqdm = tqdm(dataloader)
    dev = next(network.parameters()).device
    dtype = next(network.parameters()).dtype 
    for (i, datadict) in enumerate(t):
        with torch.no_grad():
            valid_mask = datadict["valid_mask"] 
            past_positions = datadict["past_positions"]
            past_velocities = datadict["past_velocities"]
            past_quaternions = datadict["past_quaternions"]
            future_positions = datadict["future_positions"]
            future_velocities = datadict["future_velocities"]
            tfuture = datadict["tfuture"]

            valid_past_positions : torch.Tensor = (past_positions[valid_mask].type(dtype).to(dev))[:,:,[0,2]]
            valid_past_velocities : torch.Tensor = (past_velocities[valid_mask].type(dtype).to(dev))[:,:,[0,2]]
            valid_past_quaternions : torch.Tensor = past_quaternions[valid_mask].type(dtype).to(dev)
            valid_future_positions : torch.Tensor = (future_positions[valid_mask].type(dtype).to(dev))[:,:,[0,2]]
            valid_future_velocities : torch.Tensor = (future_velocities[valid_mask].type(dtype).to(dev))[:,:,[0,2]]
            valid_tfuture : torch.Tensor = tfuture[valid_mask].type(dtype).to(dev)

            # print(valid_past_positions.shape)
            # print(valid_future_positions.shape)
            if network.input_dim==4:
                networkinput = torch.cat([valid_past_positions, valid_past_velocities], dim=2)
            elif network.input_dim==8:
                networkinput = torch.cat([valid_past_positions, valid_past_velocities, valid_past_quaternions], dim=2)
            else:
                raise ValueError("Currently, only input dimensions of 4 and 8 are supported")
            means, var_factors, covar_factors = network(networkinput)
            # print(var_factors[0])
            # print(covar_factors[0])
            curves = torch.cat([valid_future_positions[:,0].unsqueeze(1), means], dim=1)
            dt = valid_tfuture[:,-1]-valid_tfuture[:,0]
            s_torch_cur = (valid_tfuture - valid_tfuture[:,0,None])/dt[:,None]
            Mpos = mu.bezierM(s_torch_cur, network.bezier_order)
            pred_points = torch.matmul(Mpos, curves)

            idx = np.random.randint(0, high=valid_past_positions.shape[0], dtype=np.int32)
            past_pos_plot = valid_past_positions[idx].cpu().numpy()
            past_vel_plot = valid_past_velocities[idx].cpu().numpy()
            future_pos_plot = valid_future_positions[idx].cpu().numpy()
            future_vel_plot = valid_future_velocities[idx].cpu().numpy()
            pred_points_plot = pred_points[idx].cpu().numpy()

            deltas_plot = pred_points_plot - future_pos_plot
            delta_norms = np.linalg.norm(deltas_plot, ord=3, axis=1)
            maxnorm = np.max(delta_norms)
            meannorm = np.mean(delta_norms)
            print("Average Delta Norm: %f", meannorm)
            print("Max Delta Norm: %f", maxnorm)
            
            figure, ax = plt.subplots()
            plt.xlabel("X (meters)")
            plt.ylabel("Z (meters)")
            plt.scatter(past_pos_plot[:,0], past_pos_plot[:,1], label="Past Positions", c="b")
            plt.quiver(past_pos_plot[:,0], past_pos_plot[:,1], past_vel_plot[:,0], past_vel_plot[:,1], angles='xy')
            plt.plot(future_pos_plot[:,0], future_pos_plot[:,1], label="Future Positions", c="g")
            plt.quiver(future_pos_plot[:,0], future_pos_plot[:,1], future_vel_plot[:,0], future_vel_plot[:,1], angles='xy')
            plt.plot(pred_points_plot[:,0], pred_points_plot[:,1], label="Predicted Positions", c="r")
            plt.legend()
            plt.show()

        

    



if __name__=="__main__":
    main()
    