import deepracing_models.math_utils.bezier, deepracing_models.math_utils
from deepracing_models.nn_models.TrajectoryPrediction import BezierMixNet
from deepracing_models.data_loading import file_datasets as FD 
import torch, torch.optim, torch.nn.functional as F
import torch.utils.data as torchdata
import yaml
import os
import numpy as np
import pickle as pkl
from path_server.smooth_path_helper import SmoothPathHelper
import tqdm
import matplotlib.figure
import matplotlib.pyplot as plt
import json
from mix_net.src.mix_net import MixNet

# k=3
# d = 2
# num = 1

with open(os.path.join("default_configs", "bezier_mixnet.yaml"), "r") as f:
    config : dict = yaml.load(f, Loader=yaml.SafeLoader)
print(config)

timestep = 0.1
batch = 256
bezier_order = config["bezier_order"]
prediction_time = config["prediction_time"]
cuda=True
with torch.no_grad():
    garbage_leftbound : torch.Tensor = torch.randn(batch, 20, 2)
    garbage_rightbound : torch.Tensor = torch.randn_like(garbage_leftbound)
    t_samp : torch.Tensor = torch.linspace(0.0, prediction_time, steps=50).unsqueeze(0)
    s_samp : torch.Tensor = t_samp.expand(batch, t_samp.shape[1])/prediction_time
    Mbezier : torch.Tensor = deepracing_models.math_utils.bezierM(s_samp, bezier_order)

    random_curves_in = torch.linspace(-100.0, 0.0, steps=bezier_order+1).unsqueeze(0).unsqueeze(-1).expand(batch, bezier_order+1, 2).clone()
    random_curves_in[:,:-2]+=10.0*torch.randn_like(random_curves_in[:,:-2])
    random_position_history : torch.Tensor = torch.matmul(Mbezier, random_curves_in)
    Mbezierderiv, random_velocities_in_ = deepracing_models.math_utils.bezierDerivative(random_curves_in, t=s_samp, order=1)
    random_velocities_in = random_velocities_in_/prediction_time

    random_curves_out = torch.linspace(0.0, 100.0, steps=bezier_order+1).unsqueeze(0).unsqueeze(-1).expand(batch, bezier_order+1, 2).clone()
    random_curves_out[:,2:]+=10.0*torch.randn_like(random_curves_out[:,2:])
    random_labels : torch.Tensor = torch.matmul(Mbezier, random_curves_out)
    _, random_velocities_out_ = deepracing_models.math_utils.bezierDerivative(random_curves_out, M=Mbezierderiv, order=1)
    random_velocities_out = random_velocities_out_/prediction_time

    random_history : torch.Tensor = torch.cat([random_position_history, random_velocities_in], dim=-1)

with open("/home/trent/deepracingws/src/MixNet/train/configs/mix_net/net_params.json", "r") as f:
    mixnetparams = json.load(f)
net : BezierMixNet = BezierMixNet(config).float()
# mixnet : MixNet = MixNet(mixnetparams).double()
if cuda:
    net = net.cuda()
    # mixnet = mixnet.cuda()
# optimizer = torch.optim.SGD(mixnet.parameters(), lr = 1E-7, momentum = 0.0)
nesterov = False
lr = 1E-4
optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = (lr/10.0), nesterov=nesterov)

# control_points = net(random_history, garbage_leftbound, garbage_rightbound)
# print(random_curves_out - control_points)
# output = torch.matmul(Mbezier, control_points)
# loss = F.mse_loss(output, random_labels)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
# ['timestamps', 'target_positions', 'target_velocities', 'target_accelerations', 
#  'target_rotations', 'innerbound_corresponding_arclengths', 'outerbound_corresponding_arclengths',
#    'centerline_corresponding_arclengths', 'raceline_corresponding_arclengths', 'ego_positions', 
#    'ego_velocities', 'ego_accelerations', 'ego_rotations']

dsetfiles = [
    "/media/trent/T7/bags/deepracingbags/offsetlines/run1.bag_mixnet_data_bottas_odom/metadata.yaml",
    "/media/trent/T7/bags/deepracingbags/offsetlines/run1.bag_mixnet_data_hamilton_odom/metadata.yaml"
]
dsets = []
for dsetmetadatafile in dsetfiles:
    with open(dsetmetadatafile, "r") as f:
        dsetconfig = yaml.load(f, Loader=yaml.SafeLoader)
    print(dsetconfig)
    dsetdir = os.path.dirname(dsetmetadatafile)
    with open(os.path.join(dsetdir, "deepracingdata.npz"), "rb") as f:
        datadict = np.load(f, allow_pickle=True)
        print(list(datadict.keys()))
        timestamps : np.ndarray = datadict["timestamps"].copy()
        target_positions : np.ndarray = datadict["target_positions"].copy()
        target_velocities : np.ndarray = datadict["target_velocities"].copy()
        target_accelerations : np.ndarray = datadict["target_accelerations"].copy()
        target_rotations : np.ndarray = datadict["target_rotations"].copy()
        innerbound_corresponding_arclengths : np.ndarray = datadict["innerbound_corresponding_arclengths"].copy()
        outerbound_corresponding_arclengths : np.ndarray = datadict["outerbound_corresponding_arclengths"].copy()
        centerline_corresponding_arclengths : np.ndarray = datadict["centerline_corresponding_arclengths"].copy()
        raceline_corresponding_arclengths : np.ndarray = datadict["raceline_corresponding_arclengths"].copy()
    with open(os.path.join(dsetdir, "scipy_line_helpers.pkl"), "rb") as f:
        helpersdict : dict[str, SmoothPathHelper] = pkl.load(f)
    dset : FD.TrajectoryPredictionDataset = FD.TrajectoryPredictionDataset(target_positions,
                                                                        target_velocities,
                                                                        helpersdict["inner_bound"],
                                                                        innerbound_corresponding_arclengths,
                                                                        helpersdict["outer_bound"],
                                                                        outerbound_corresponding_arclengths,
                                                                        helpersdict["center_line"],
                                                                        centerline_corresponding_arclengths,
                                                                        helpersdict["optimal_line"],
                                                                        outerbound_corresponding_arclengths
                                                                        )
    dsets.append(dset)
import time
concatdset : torchdata.ConcatDataset = torchdata.ConcatDataset(dsets)
dataloader = torchdata.DataLoader(concatdset, batch_size=batch, pin_memory=True)
net.train()
for asdf in range(200):
    totalloss = 0.0
    n = 0
    tq = tqdm.tqdm(enumerate(dataloader), desc="Yay")
    for (i, batchdict) in tq:
        position_history : torch.Tensor = (batchdict["target_position_history"].cuda() if cuda else batchdict["target_position_history"]).float()
        velocity_history : torch.Tensor = (batchdict["target_velocity_history"].cuda() if cuda else batchdict["target_velocity_history"]).float()
        position_future : torch.Tensor = (batchdict["target_position_future"].cuda() if cuda else batchdict["target_position_future"]).float()
        left_bound : torch.Tensor = (batchdict["outer_boundary_input"].cuda() if cuda else batchdict["outer_boundary_input"]).float()
        right_bound : torch.Tensor = (batchdict["inner_boundary_input"].cuda() if cuda else batchdict["inner_boundary_input"]).float()
        history : torch.Tensor = torch.cat([position_history, velocity_history], dim=-1)
        currentbatchsize = position_history.shape[0]

        inner_boundary_label : torch.Tensor = (batchdict["inner_boundary_label"].cuda() if cuda else batchdict["inner_boundary_label"]).float()
        outer_boundary_label : torch.Tensor = (batchdict["outer_boundary_label"].cuda() if cuda else batchdict["outer_boundary_label"]).float()
        centerline_label : torch.Tensor = (batchdict["centerline_label"].cuda() if cuda else batchdict["centerline_label"]).float()
        raceline_label : torch.Tensor = (batchdict["raceline_label"].cuda() if cuda else batchdict["raceline_label"]).float()

        # mixingratios, vel_out, acc_out = mixnet(position_history, left_bound, right_bound)
        
        # predicted_position_future : torch.Tensor = inner_boundary_label*(mixingratios[:,0])[:,None,None]+\
        #                                            outer_boundary_label*(mixingratios[:,1])[:,None,None]+\
        #                                            centerline_label*(mixingratios[:,2])[:,None,None]+\
        #                                            raceline_label*(mixingratios[:,3])[:,None,None]

        extra_control_points : torch.Tensor = net(position_history, velocity_history, left_bound, right_bound)
        tsamp : torch.Tensor = torch.linspace(0.0, net.prediction_time, steps=position_future.shape[1], dtype=extra_control_points.dtype, device=extra_control_points.device)
        
        p0 : torch.Tensor = position_future[:,0]
        # initial_velocity : torch.Tensor = velocity_history[:,-1]
        # p1 : torch.Tensor = p0 + net.prediction_time*initial_velocity/net.bezier_order
        control_points = torch.cat([p0.unsqueeze(-2), extra_control_points], dim=-2)
        ssamp : torch.Tensor = (tsamp/tsamp[-1]).unsqueeze(0)
        Msamp = deepracing_models.math_utils.bezierM(ssamp.expand(currentbatchsize, ssamp.shape[1]), net.bezier_order)
        predicted_position_future = torch.matmul(Msamp, control_points)

        position_history_cpu = position_history[0].cpu()
        position_future_cpu = position_future[0].cpu()
        predicted_position_future_cpu = predicted_position_future[0].detach().cpu()
        left_bound_cpu = left_bound[0].cpu()
        right_bound_cpu = right_bound[0].cpu()

        

        loss : torch.Tensor = torch.mean(torch.square(predicted_position_future - position_future))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totalloss += loss.item()
        n+=1
        averageloss = totalloss/n
        tq.set_postfix({"average loss" : averageloss})
        # time.sleep(0.15)
    # fig : matplotlib.figure.Figure = plt.figure()
    # plt.plot(position_history_cpu[:,0], position_history_cpu[:,1], label="Position History")
    # plt.plot(position_future_cpu[:,0], position_future_cpu[:,1], label="Ground Truth Future")
    # plt.plot(predicted_position_future_cpu[:,0], predicted_position_future_cpu[:,1], label="Prediction")
    # plt.plot(left_bound_cpu[:,0], left_bound_cpu[:,1], label="Left Bound")
    # plt.plot(right_bound_cpu[:,0], right_bound_cpu[:,1], label="Right Bound")
    # plt.legend()
    # plt.axis("equal")
    # plt.show()

