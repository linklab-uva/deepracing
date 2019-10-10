import comet_ml
import torch
import torch.nn as NN
import torch.utils.data as data_utils
import data_loading.proto_datasets
from tqdm import tqdm as tqdm
import nn_models.LossFunctions as loss_functions
import nn_models.Models
import numpy as np
import torch.optim as optim
from tqdm import tqdm as tqdm
import pickle
from datetime import datetime
import os
import string
import argparse
import scipy.integrate
import torchvision.transforms as transforms
import yaml
import shutil
import skimage
import skimage.io
import deepracing.backend
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import scipy.interpolate
import numpy.linalg as la
import math_utils.bezier
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
def run_epoch(network, testLoader, gpu, loss_func, imsize=(66,200), debug=False,  use_float=True):
    lossfloat = 0.0
    batch_size = testLoader.batch_size
    num_samples=0.0
    splK = 3
    numpoints = 11
    t : tqdm = tqdm(enumerate(testLoader), total=len(testLoader))
    network.eval()  # This is important to call before testing!
    losses=np.zeros(len(testLoader.dataset))
    gterrors=np.zeros(len(testLoader.dataset))
    gtvelerrors=np.zeros(len(testLoader.dataset))
    for (i, (image_torch, opt_flow_torch, positions_torch, quats_torch, linear_velocities_torch, angular_velocities_torch, session_times_torch) ) in t:
        if network.input_channels==5:
            image_torch = torch.cat((image_torch,opt_flow_torch),axis=2)
        if use_float:
            image_torch = image_torch.float()
            positions_torch = positions_torch.float()
            linear_velocities_torch = linear_velocities_torch.float()
            session_times_torch = session_times_torch.float()
        else:
            image_torch = image_torch.double()
            positions_torch = positions_torch.double()
            session_times_torch = session_times_torch.double()
            linear_velocities_torch = linear_velocities_torch.double()
        if gpu>=0:
            image_torch = image_torch.cuda(gpu)
            positions_torch = positions_torch.cuda(gpu)
            session_times_torch = session_times_torch.cuda(gpu)
            linear_velocities_torch = linear_velocities_torch.cuda(gpu)
 
        predictions = network(image_torch)
        predictions_reshape = predictions.transpose(1,2)
        dt = session_times_torch[:,-1]-session_times_torch[:,0]
        s_torch = (session_times_torch - session_times_torch[:,0,None])/dt[:,None]
        fitpoints = positions_torch[:,:,[0,2]]
        fitvels = linear_velocities_torch[:,:,[0,2]]
        bezier_order = network.params_per_dimension-1
        Mfit, controlpoints_fit = math_utils.bezier.bezierLsqfit(fitpoints,s_torch,bezier_order)
        gtevalpoints = torch.matmul(Mfit, controlpoints_fit)
        _, gt_fit_vels = math_utils.bezier.bezierDerivative(controlpoints_fit,s_torch)
        pred_eval_points = torch.matmul(Mfit, predictions_reshape)
        _, pred_vels = math_utils.bezier.bezierDerivative(predictions_reshape,s_torch)
        loss = torch.mean( torch.sqrt(loss_func(pred_eval_points,fitpoints)),dim=1)
       # print(loss.shape)

        currloss = float(torch.sum(loss).item())
        lossfloat+=currloss
        if testLoader.batch_size==1:
            losses[i]=currloss
            gterrors[i]=torch.mean(torch.norm(fitpoints-gtevalpoints,p=2,dim=2)).item()
            gtvelerrors[i]=torch.mean(torch.norm(fitvels-gt_fit_vels/dt[:,None,None],p=2,dim=2)).item()
        else:
            istart = i*testLoader.batch_size
            iend = istart+s_torch.shape[0]
            losses[istart:iend]=loss.detach().cpu().numpy()
            gterrors[istart:iend]=torch.mean(torch.norm(fitpoints-gtevalpoints,p=2,dim=2),dim=1).detach().cpu().numpy()
            gtvelerrors[istart:iend]=torch.mean(torch.norm(fitvels-gt_fit_vels/dt[:,None,None],p=2,dim=2),dim=1).detach().cpu().numpy()
        num_samples += testLoader.batch_size
        if debug:
            print("Current loss: %s" %(str(loss)))
            predevalpoints = torch.matmul(Mfit, predictions_reshape)
            xprednp = predevalpoints[0,:,0].cpu().detach().numpy()
            zprednp = predevalpoints[0,:,1].cpu().detach().numpy()
            xvelprednp = pred_vels[0,:,0].cpu().detach().numpy()
            zvelprednp = pred_vels[0,:,1].cpu().detach().numpy()
            pred_control_points_np = predictions_reshape[0].cpu().detach().numpy()
            gt_control_points_np = controlpoints_fit[0].cpu().numpy()
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(fitpoints[0,:,0].cpu().numpy(),fitpoints[0,:,1].cpu().numpy(),'r-')
           # ax.plot(gt_control_points_np[:,0],gt_control_points_np[:,1],'go')
            ax.scatter(gtevalpoints[0,:,0].cpu().numpy(),gtevalpoints[0,:,1].cpu().numpy(), facecolors='none', edgecolors='g')

            ax.scatter(xprednp,zprednp, facecolors='none', edgecolors='b')
            #ax.scatter(pred_control_points_np[:,0],pred_control_points_np[:,1], facecolors='none', edgecolors='r')
            skipn = 1
            #ax.quiver(gtevalpoints[0,::skipn,0].cpu().numpy(),gtevalpoints[0,::skipn,1].cpu().numpy(),(100/60)*gt_fit_vels[0,::skipn,0].cpu().numpy(),gt_fit_vels[0,::skipn,1].cpu().numpy())

            # zmax = np.max(zprednp) + 10
            # xmin = np.min(xprednp) - 5
            # xmax = np.max(xprednp) + 5
            # dx = xmax-xmin
            # plt.xlim(xmin,xmax)
            # plt.ylim(0,zmax)
            plt.show()
        t.set_postfix({"posloss":lossfloat/num_samples})
    return losses, gterrors, gtvelerrors

def plotStatistics(errors, label):
    figkde, axkde = plt.subplots()
    figkde.subplots_adjust(hspace=0.05, wspace=0.05)
    kde = KernelDensity(bandwidth=0.1).fit(errors.reshape(-1, 1))

    distancesplot = np.linspace(0,np.max(errors),2*distances.shape[0])# + 1.5*stddist)
    kdexplot = distancesplot.reshape(-1, 1)
    log_dens = kde.score_samples(kdexplot)
    pdf = np.exp(log_dens)
    axkde.plot(np.hstack((np.array([0]),distancesplot)), np.hstack((np.array([0]),pdf)), '-')
    axkde.set_xlabel("Distance from prediction to ground truth %s" %(label))
    axkde.set_ylabel("Probability Density (pdf)")

    
    figcdf, axcdf = plt.subplots()

    cdf = distance_indices/distance_indices.shape[0]
    axcdf.plot(distances, cdf, '-')
    axcdf.set_xlabel("Distance from prediction to ground truth")
    axcdf.set_ylabel("Cumulative Probability Density (cdf)")

def go():
    parser = argparse.ArgumentParser(description="Train AdmiralNet Pose Predictor")
    parser.add_argument("model_file", type=str,  help="Weight file to load")
    parser.add_argument("dataset_config_file", type=str,  help="Dataset config file to load")

    parser.add_argument("--gpu", type=int, default=-1,  help="GPU to use")
    parser.add_argument("--debug", action="store_true",  help="Display images upon each iteration of the training loop")
    args = parser.parse_args()
    dataset_config_file = args.dataset_config_file
    model_file = args.model_file
    config_file = os.path.join(os.path.dirname(model_file),"config.yaml")
    comet_config_file = os.path.join(os.path.dirname(model_file),"experiment_config.yaml")
    debug = args.debug
    with open(config_file,'r') as f:
        config = yaml.load(f, Loader = yaml.SafeLoader)
    with open(dataset_config_file,'r') as f:
        datasets_config = yaml.load(f, Loader = yaml.SafeLoader)
    with open(comet_config_file,'r') as f:
        comet_config = yaml.load(f, Loader = yaml.SafeLoader)

    image_size = datasets_config["image_size"]
    hidden_dimension = config["hidden_dimension"]
    input_channels = config["input_channels"]
    context_length = config["context_length"]
    bezier_order = config["bezier_order"]
    gpu = args.gpu
    loss_weights = config["loss_weights"]
    temporal_conv_feature_factor = config["temporal_conv_feature_factor"]
    debug = args.debug
    use_float = config["use_float"]
    learnable_initial_state = config.get("learnable_initial_state",True)
    print("Using config:\n%s" % (str(config)))
    net = nn_models.Models.AdmiralNetCurvePredictor(context_length= context_length, input_channels=input_channels, params_per_dimension=bezier_order+1) 
    net.load_state_dict(torch.load(model_file,map_location=torch.device("cpu")))
    print("net:\n%s" % (str(net)))

    loss_func = nn_models.LossFunctions.SquaredLpNormLoss(time_reduction="oogabooga",batch_reduction="oogabooga")
    if use_float:
        print("casting stuff to float")
        net = net.float()
        loss_func = loss_func.float()
    else:
        print("casting stuff to double")
        net = net.double()
        loss_func = loss_func.double()
    if gpu>=0:
        print("moving stuff to GPU")
        net = net.cuda(gpu)
        loss_func = loss_func.cuda(gpu)
    
    
    num_workers = 0
    max_spare_txns = 50
    #image_wrapper = deepracing.backend.ImageFolderWrapper(os.path.dirname(image_db))
    dsets=[]
    use_optflow = net.input_channels==5
    for dataset in datasets_config["datasets"]:
        print("Parsing database config: %s" %(str(dataset)))
        label_folder = dataset["label_folder"]
        key_file = dataset["key_file"]
        image_folder = dataset["image_folder"]
        apply_color_jitter = dataset.get("apply_color_jitter",False)
        erasing_probability = dataset.get("erasing_probability",0.0)
        label_wrapper = deepracing.backend.PoseSequenceLabelLMDBWrapper()
        label_wrapper.readDatabase(os.path.join(label_folder,"lmdb"), max_spare_txns=max_spare_txns )
        image_mapsize = float(np.prod(image_size)*3+12)*float(len(label_wrapper.getKeys()))*1.1

        image_wrapper = deepracing.backend.ImageLMDBWrapper(direct_caching=False)
        image_wrapper.readDatabase(os.path.join(image_folder,"image_lmdb"), max_spare_txns=max_spare_txns, mapsize=image_mapsize )


        curent_dset = data_loading.proto_datasets.PoseSequenceDataset(image_wrapper, label_wrapper, key_file, context_length, image_size = image_size, return_optflow=use_optflow,\
            apply_color_jitter=False, erasing_probability=0.0, geometric_variants = False)
        dsets.append(curent_dset)
        print("\n")
    if len(dsets)==1:
        dset = dsets[0]
    else:
        dset = torch.utils.data.ConcatDataset(dsets)
    
    dataloader = data_utils.DataLoader(dset)
    print("Dataloader of of length %d" %(len(dataloader)))
    losses, gterrors, gtvelerrors = run_epoch(net, dataloader, gpu, loss_func, debug=debug, use_float = use_float)
    #distances = np.sqrt(losses)
    distance_sort = np.argsort(losses)
    distances = losses.copy()[distance_sort]
    distance_indices = np.linspace(0,distances.shape[0]-1,distances.shape[0]).astype(np.float32)

    distance_sort_descending = np.flip(distance_sort.copy())
    distances_sorted_descending = losses.copy()[distance_sort_descending]
    meandist = np.mean(distances)
    stddist = np.std(distances)
    print(gterrors)
    print("RMS gt fit error: %f" % ( np.mean(gterrors) ) )
    print(gtvelerrors)
    print("RMS gt fit vel error: %f" % ( np.mean(gtvelerrors) ) )
    print(distances)
    print("RMS error: %f" % (  np.mean(losses) ) ) 
   # plt.plot(distance_indices,distances)
    
    figkde, axkde = plt.subplots()
    figkde.subplots_adjust(hspace=0.05, wspace=0.05)
    kde = KernelDensity(bandwidth=0.1).fit(distances.reshape(-1, 1))

    distancesplot = np.linspace(0,np.max(distances),2*distances.shape[0])# + 1.5*stddist)
    kdexplot = distancesplot.reshape(-1, 1)
    log_dens = kde.score_samples(kdexplot)
    pdf = np.exp(log_dens)
    axkde.plot(np.hstack((np.array([0]),distancesplot)), np.hstack((np.array([0]),pdf)), '-')
    axkde.set_xlabel("Distance from prediction to ground truth")
    axkde.set_ylabel("Probability Density (pdf)")

    
    figcdf, axcdf = plt.subplots()

    cdf = distance_indices/distance_indices.shape[0]
    axcdf.plot(distances, cdf, '-')
    axcdf.set_xlabel("Distance from prediction to ground truth")
    axcdf.set_ylabel("Cumulative Probability Density (cdf)")
    try:
        plt.show()
    except:
        pass
    bezier_order = net.params_per_dimension-1
    try:
        experiment =  comet_ml.ExistingExperiment(workspace="electric-turtle", project_name="deepracingbezierpredictor", previous_experiment=comet_config["experiment_key"])
        with experiment.test():
            for i, idx in enumerate(distance_sort_descending[0:100]):
                print("Running image %d"%(idx))
                image_torch, opt_flow_torch, positions_torch, quats_torch, linear_velocities_torch, angular_velocities_torch, session_times_torch = dset[idx]
                image_numpy = np.round(255.0*(image_torch.numpy().copy().transpose(0,2,3,1))).astype(np.uint8)
                predictions_reshape = net(image_torch.double().cuda(gpu).unsqueeze(0)).transpose(1,2)
                last_image = image_numpy[-1]
                fig = plt.figure()
                ax_im = fig.add_subplot(121)
                ax_im.imshow(last_image)
                ax_pos = fig.add_subplot(122)
                ax_pos.set_xlim(-15, 15)
                ax_pos.set_title("Predicted and Ground-Truth Paths")
                ax_pos.set_xlabel("X Position In Local Coordinates (meters)")
                ax_pos.set_ylabel("Z Position In Local Coordinates (meters)")
                dt = session_times_torch[-1]-session_times_torch[0]
                s_torch = ((session_times_torch - session_times_torch[0])/dt).double().cuda(gpu)
                fitpoints = positions_torch[:,[0,2]].double().cuda(gpu)
                M = math_utils.bezierM(s_torch.unsqueeze(0),bezier_order)
                pred_eval_points = torch.matmul(M, predictions_reshape)[0].cpu().detach().numpy().copy()
                positions_numpy = fitpoints.cpu().detach().numpy().copy()

                ax_pos.plot(-pred_eval_points[:,0],pred_eval_points[:,1], label='Predictions', color='r')
                ax_pos.scatter(-pred_eval_points[:,0],pred_eval_points[:,1], facecolors='none', edgecolors='r')

                ax_pos.plot(-positions_numpy[:,0],positions_numpy[:,1], label='Ground Truth', color='g')
                ax_pos.scatter(-positions_numpy[:,0],positions_numpy[:,1], facecolors='none', edgecolors='g')
                ax_pos.set_aspect("equal")
                ax_pos.legend()

                experiment.log_figure(figure_name="image_%d"%(i), figure=fig , overwrite=True )
    except KeyboardInterrupt as key:
        pass


import logging
if __name__ == '__main__':
    logging.basicConfig()
    go()    
    