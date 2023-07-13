import deepracing_models.math_utils.bezier, deepracing_models.math_utils
from deepracing_models.nn_models.TrajectoryPrediction import BezierMixNet
from deepracing_models.data_loading import file_datasets as FD 
import torch, torch.optim, torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
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
from scipy.spatial.transform import Rotation
from mix_net.src.mix_net_dataset import MixNetDataset
import shutil
# k=3
# d = 2
# num = 1

thisfiledir = os.path.dirname(__file__)
with open(os.path.join(thisfiledir, "default_configs", "bezier_mixnet.yaml"), "r") as f:
    netconfig : dict = yaml.load(f, Loader=yaml.SafeLoader)
print(netconfig)



bagsdir = "/p/DeepRacing/deepracingbags/offsetlines"
dsetfiles = [
    os.path.join(bagsdir, "run1.bag_mixnet_data_hamilton_odom/metadata.yaml"),
    os.path.join(bagsdir, "run1.bag_mixnet_data_bottas_odom/metadata.yaml"),
]
graphics_dir = os.path.join(bagsdir, "graphics")
if os.path.isdir(graphics_dir):
    shutil.rmtree(graphics_dir)
os.makedirs(graphics_dir)
dsets = []
dsetconfigs = []
numsamples_prediction = None
for dsetmetadatafile in dsetfiles:
    with open(dsetmetadatafile, "r") as f:
        dsetconfig = yaml.load(f, Loader=yaml.SafeLoader)
    if numsamples_prediction is None:
        numsamples_prediction = dsetconfig["numsamples_prediction"]
    elif numsamples_prediction!=dsetconfig["numsamples_prediction"]:
        raise ValueError("All datasets must have the same number of prediction points. " + \
                         "Dataset at %s has prediction length %d, but previous dataset " + \
                         "has prediction length %d" % (dsetmetadatafile, dsetconfig["numsamples_prediction"], numsamples_prediction))
    dsetconfigs.append(dsetconfig)
    dsetdir = os.path.dirname(dsetmetadatafile)
    labelfile = dsetconfig["train_data"]
    with open(os.path.join(dsetdir, labelfile), "rb") as f:
        data = pkl.load(f)
    dsets.append(MixNetDataset(data, 0.0, 31, dsetconfig["trackname"]))
import time
timestep = 0.1
batch = 64
# bezier_order = config["bezier_order"]
# prediction_time = config["prediction_time"]
cuda=True
dataloader = torchdata.DataLoader(torchdata.ConcatDataset(dsets), batch_size=batch, pin_memory=cuda, shuffle=True)#, collate_fn=dsets[0].collate_fn)
net : BezierMixNet = BezierMixNet(netconfig).float()
lossfunc = torch.nn.MSELoss().float()
if cuda:
    net = net.cuda()
    lossfunc = lossfunc.cuda()
firstparam = next(net.parameters())
dtype = firstparam.dtype
device = firstparam.device
nesterov = True
lr = 5E-4
# momentum = lr/10.0
# optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = momentum, nesterov=(nesterov and (momentum>0.0)))
optimizer = torch.optim.Adam(net.parameters(), lr = lr)#, weight_decay=1e-8)
scheduler = ExponentialLR(optimizer, 0.9999)


num_accel_sections : int = netconfig["acc_decoder"]["num_acc_sections"]
_time_profile_matrix = torch.zeros( (numsamples_prediction-1, num_accel_sections), dtype=dtype, device=device)
ratio = int(numsamples_prediction/num_accel_sections)
prediction_timestep = dsetconfigs[0]["timestep_prediction"]
prediction_totaltime = dsetconfigs[0]["predictiontime"]
tf = prediction_totaltime/num_accel_sections
for i in range(num_accel_sections):
    _time_profile_matrix[(i * ratio) : ((i + 1) * ratio), i] = torch.linspace(
        prediction_timestep, tf, ratio, dtype=dtype, device=device
    )
    _time_profile_matrix[((i + 1) * ratio) :, i] = tf
_time_profile_matrix = torch.cat([torch.zeros_like(_time_profile_matrix[0]).unsqueeze(0), _time_profile_matrix], dim=0)
_time_profile_matrix = _time_profile_matrix.unsqueeze(0)
print(_time_profile_matrix)

# print(_time_profile_matrix)
# print(_time_profile_matrix.shape)
# time.sleep(2.0)
net.train()
averageloss = 1E9
averagepositionloss = 1E9
averagevelocityloss = 1E9
averagevelocityerror = 1E9
epoch = 0
fignumber = 1

kbezier = 4
while (averagepositionloss>1.0) or (averagevelocityerror>1.0):
    totalloss = 0.0
    total_position_loss = 0.0
    total_velocity_loss = 0.0
    total_velocity_error = 0.0
    epoch+=1
    tq = tqdm.tqdm(enumerate(dataloader), desc="Yay")
    for (i, datatuple) in tq:
        position_history, position_future, tangent_future, speed_future, fut_inds_batch, future_arclength, \
            left_bound_input, right_bound_input, \
                left_boundary_label, right_boundary_label, \
                    centerline_label, raceline_label, \
                        left_boundary_label_arclength, right_boundary_label_arclength, \
                            centerline_label_arclength, raceline_label_arclength, \
                                bcurves_r, tracknames = datatuple
        if cuda:
            position_history = position_history.cuda().type(dtype)
            position_future = position_future.cuda().type(dtype)
            speed_future = speed_future.cuda().type(dtype)
            future_arclength = future_arclength.cuda().type(dtype)
            left_bound_input = left_bound_input.cuda().type(dtype)
            right_bound_input = right_bound_input.cuda().type(dtype)
            left_boundary_label = left_boundary_label.cuda().type(dtype)
            right_boundary_label = right_boundary_label.cuda().type(dtype)
            centerline_label = centerline_label.cuda().type(dtype)
            raceline_label = raceline_label.cuda().type(dtype)
            left_boundary_label_arclength = left_boundary_label_arclength.cuda().type(dtype)
            right_boundary_label_arclength = right_boundary_label_arclength.cuda().type(dtype)
            centerline_label_arclength = centerline_label_arclength.cuda().type(dtype)
            raceline_label_arclength = raceline_label_arclength.cuda().type(dtype)
            bcurves_r = bcurves_r.cuda().type(dtype)
            tangent_future = tangent_future.cuda().type(dtype)
        else:
            position_history = position_history.cpu().type(dtype)
            position_future = position_future.cpu().type(dtype)
            speed_future = speed_future.cpu().type(dtype)
            left_bound_input = left_bound_input.cpu().type(dtype)
            future_arclength = future_arclength.cpu().type(dtype)
            right_bound_input = right_bound_input.cpu().type(dtype)
            left_boundary_label = left_boundary_label.cpu().type(dtype)
            right_boundary_label = right_boundary_label.cpu().type(dtype)
            centerline_label = centerline_label.cpu().type(dtype)
            raceline_label = raceline_label.cpu().type(dtype)
            left_boundary_label_arclength = left_boundary_label_arclength.cpu().type(dtype)
            right_boundary_label_arclength = right_boundary_label_arclength.cpu().type(dtype)
            centerline_label_arclength = centerline_label_arclength.cpu().type(dtype)
            raceline_label_arclength = raceline_label_arclength.cpu().type(dtype)
            bcurves_r = bcurves_r.cpu().type(dtype)
            tangent_future = tangent_future.cpu().type(dtype)
        currentbatchsize = int(position_history.shape[0])

        delta_r : torch.Tensor = future_arclength[:,-1] - future_arclength[:,0]
        future_arclength_rel : torch.Tensor = future_arclength - future_arclength[:,0,None]

        known_control_points : torch.Tensor = torch.zeros_like(bcurves_r[:,0,:2])
        known_control_points[:,0] = position_future[:,0]
        known_control_points[:,1] = known_control_points[:,0] + (delta_r[:,None]/kbezier)*tangent_future[:,0]

        # print(tangent_future[:,0])
        (mix_out, acc_out) = net(position_history, left_bound_input, right_bound_input)
        mixed_control_points = torch.sum(bcurves_r[:,:,2:]*mix_out[:,:,None,None], dim=1)# + known_control_points[:,1,None]

        predicted_bcurve = torch.cat([known_control_points, mixed_control_points], dim=1) 

        time_profile_matrices = _time_profile_matrix.expand(currentbatchsize, _time_profile_matrix.shape[1], _time_profile_matrix.shape[2])
        speed_profile_out = torch.ones_like(speed_future)*speed_future[:,0,None] + (time_profile_matrices @ acc_out.unsqueeze(-1)).squeeze(-1)
        
        accel_profile_out = (speed_profile_out[:,1:] - speed_profile_out[:,:-1])*prediction_timestep
        dr_out = prediction_timestep*speed_profile_out[:,:-1] + (0.5*(prediction_timestep**2))*accel_profile_out
        arclengths_out = torch.zeros_like(future_arclength)
        arclengths_out[:,1:] = torch.cumsum(dr_out, 1)
        arclengths_out_s = arclengths_out/(arclengths_out[:,-1,None])

        # arclengths_out_s = future_arclength_rel/delta_r[:,None]
        Mbezierout = deepracing_models.math_utils.bezierM(arclengths_out_s, kbezier)

        predicted_position_future = torch.matmul(Mbezierout, predicted_bcurve)


        
        loss_position : torch.Tensor = (lossfunc(predicted_position_future, position_future))
        loss_velocity : torch.Tensor = (prediction_timestep**2)*(lossfunc(speed_profile_out, speed_future))
        arclength_loss : torch.Tensor = 0.1*F.mse_loss(arclengths_out, future_arclength_rel)
        loss = loss_position + loss_velocity + arclength_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_position_loss += loss_position.item()
        total_velocity_loss += loss_velocity.item()
        total_velocity_error += loss_velocity.item()/(prediction_timestep**2)
        totalloss += loss.item()
        averageloss = totalloss/(i+1)
        averagepositionloss = total_position_loss/(i+1)
        averagevelocityloss = total_velocity_loss/(i+1)
        averagevelocityerror = total_velocity_error/(i+1)
        tq.set_postfix({"average position loss" : averagepositionloss, "average velocity error" : averagevelocityerror, "average velocity loss" : averagevelocityloss, "average loss" : averageloss, "epoch": epoch})
    if (averagepositionloss<9.0) and (epoch%25)==0:
        position_history_cpu = position_history[0].cpu()
        position_future_cpu = position_future[0].cpu()
        predicted_position_future_cpu = predicted_position_future[0].detach().cpu()
        left_bound_input_cpu = left_bound_input[0].cpu()
        right_bound_input_cpu = right_bound_input[0].cpu()
        left_bound_label_cpu = left_boundary_label[0].cpu()
        right_bound_label_cpu = right_boundary_label[0].cpu()
        centerline_label_cpu = centerline_label[0].cpu()
        raceline_label_cpu = raceline_label[0].cpu()
        fig : matplotlib.figure.Figure = plt.figure()
        scale_array = 0.5
        plt.plot(position_history_cpu[:,0], position_history_cpu[:,1], label="Position History")#, s=scale_array)
        plt.plot(position_history_cpu[0,0], position_history_cpu[0,1], "g*", label="Position History Start")
        plt.plot(position_history_cpu[-1,0], position_history_cpu[-1,1], "r*", label="Position History End")
        plt.scatter(position_future_cpu[:,0], position_future_cpu[:,1], label="Ground Truth Future", s=scale_array)
        plt.plot(predicted_position_future_cpu[:,0], predicted_position_future_cpu[:,1], label="Prediction")#, s=scale_array)
        plt.plot(centerline_label_cpu[:,0], centerline_label_cpu[:,1], label="Centerline Label")#, s=scale_array)
        plt.plot(raceline_label_cpu[:,0], raceline_label_cpu[:,1], label="Raceline Label")#, s=scale_array)
        plt.plot([],[], label="Boundaries", color="navy")#, s=scale_array)
        plt.legend()
        # plt.plot(left_bound_input_cpu[:,0], left_bound_input_cpu[:,1], label="Left Bound Input", color="navy")
        # plt.plot(right_bound_input_cpu[:,0], right_bound_input_cpu[:,1], label="Right Bound Input", color="navy")
        plt.plot(left_bound_label_cpu[:,0], left_bound_label_cpu[:,1], label="Left Bound Label", color="navy")#, s=scale_array)
        plt.plot(right_bound_label_cpu[:,0], right_bound_label_cpu[:,1], label="Right Bound Label", color="navy")#, s=scale_array)
        plt.axis("equal")
        print(mix_out)
        fig.savefig(os.path.join(graphics_dir, "fig_%d.png" % fignumber))
        fig.savefig(os.path.join(graphics_dir, "fig_%d.svg" % fignumber))
        fig.savefig(os.path.join(graphics_dir, "fig_%d.pdf" % fignumber))
        with open(os.path.join(graphics_dir, "fig_%d.pkl" % fignumber), "wb") as f:
            pkl.dump(fig, f)
        plt.close(fig=fig)
        fignumber+=1
        # plt.show()
    # scheduler.step()

