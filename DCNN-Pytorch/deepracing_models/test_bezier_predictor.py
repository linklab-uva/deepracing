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
def run_epoch(network, testLoader, gpu, loss_func, imsize=(66,200), debug=False,  use_float=True):
    lossfloat = 0.0
    batch_size = testLoader.batch_size
    num_samples=0.0
    splK = 3
    numpoints = 11
    t : tqdm = tqdm(enumerate(testLoader), total=len(testLoader))
    network.eval()  # This is important to call before testing!
    losses=np.zeros(len(testLoader))
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
        gt_fit_vels = math_utils.bezier.bezierDerivative(controlpoints_fit,bezier_order,s_torch)
        pred_eval_points = torch.matmul(Mfit, predictions_reshape)
        pred_vels = math_utils.bezier.bezierDerivative(predictions_reshape,bezier_order,s_torch)
        loss = loss_func(pred_eval_points,fitpoints)

        currloss = float(loss.item())
        lossfloat+=currloss
        losses[i]=currloss
        num_samples += 1.0
        if debug:
            gtevalpoints = torch.matmul(Mfit, controlpoints_fit)
            predevalpoints = torch.matmul(Mfit, predictions_reshape)
            xprednp = predevalpoints[0,:,0].cpu().detach().numpy()
            zprednp = predevalpoints[0,:,1].cpu().detach().numpy()
            xvelprednp = pred_vels[0,:,0].cpu().detach().numpy()
            zvelprednp = pred_vels[0,:,1].cpu().detach().numpy()
            pred_control_points_np = predictions_reshape[0].cpu().detach().numpy()
            gt_control_points_np = controlpoints_fit[0].cpu().numpy()
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(positions_torch[0,:,0].cpu().numpy(),positions_torch[0,:,2].cpu().numpy(),'g-')
            #ax.plot(gtevalpoints[0,:,0].cpu().numpy(),gtevalpoints[0,:,1].cpu().numpy(),'r-')
            ax.plot(gt_control_points_np[:,0],gt_control_points_np[:,1],'go')

            ax.plot(xprednp,zprednp,'b-')
            ax.plot(pred_control_points_np[:,0],pred_control_points_np[:,1],'bo')
            skipn = 1
            #ax.quiver(gtevalpoints[0,::skipn,0].cpu().numpy(),gtevalpoints[0,::skipn,1].cpu().numpy(),(100/60)*gt_fit_vels[0,::skipn,0].cpu().numpy(),gt_fit_vels[0,::skipn,1].cpu().numpy())

            zmax = np.max(zprednp) + 10
            xmin = np.min(xprednp) - 5
            xmax = np.max(xprednp) + 5
            dx = xmax-xmin
            #ax.quiver(xprednp[::skipn],zprednp[::skipn],(zmax/dx)*xvelprednp[::skipn],zvelprednp[::skipn])
            plt.xlim(xmin,xmax)
            plt.ylim(0,zmax)
            #ax.plot(bezier_control_points[i,:,0].numpy(),bezier_control_points[i,:,1].numpy(),'go')
            plt.show()
        t.set_postfix({"posloss":lossfloat/num_samples})
    return losses
def go():
    parser = argparse.ArgumentParser(description="Train AdmiralNet Pose Predictor")
    parser.add_argument("config_file", type=str,  help="Configuration file to load")
    parser.add_argument("model_file", type=str,  help="Weight file to load")

    parser.add_argument("--gpu", type=int, default=-1,  help="GPU to use")
    parser.add_argument("--debug", action="store_true",  help="Display images upon each iteration of the training loop")
    args = parser.parse_args()
    config_file = args.config_file
    model_file = args.model_file
    debug = args.debug
    with open(config_file) as f:
        config = yaml.load(f, Loader = yaml.SafeLoader)

    image_size = config["image_size"]
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
    net = nn_models.Models.AdmiralNetCurvePredictor(input_channels=input_channels, params_per_dimension=bezier_order+1) 
    net.load_state_dict(torch.load(model_file,map_location=torch.device("cpu")))
    print("net:\n%s" % (str(net)))

    loss_func = torch.nn.MSELoss(reduction="mean")
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
    datasets = config["datasets"]
    dsets=[]
    use_optflow=True
    for dataset in datasets:
        print("Parsing database config: %s" %(str(dataset)))
        image_db = dataset["image_db"]
        opt_flow_db = dataset.get("opt_flow_db", "")
        label_db = dataset["label_db"]
        key_file = dataset["key_file"]
        label_wrapper = deepracing.backend.PoseSequenceLabelLMDBWrapper()
        label_wrapper.readDatabase(label_db, max_spare_txns=max_spare_txns )
        image_size = np.array(image_size)
        image_mapsize = float(np.prod(image_size)*3+12)*float(len(label_wrapper.getKeys()))*1.1
        image_wrapper = deepracing.backend.ImageLMDBWrapper(direct_caching=False)
        image_wrapper.readDatabase(image_db, max_spare_txns=max_spare_txns, mapsize=image_mapsize )
        optical_flow_db_wrapper = None
        if not opt_flow_db=='':
            print("Using optical flow database at %s" %(opt_flow_db))
            optical_flow_db_wrapper = deepracing.backend.OpticalFlowLMDBWrapper()
            optical_flow_db_wrapper.readDatabase(opt_flow_db, max_spare_txns=max_spare_txns, mapsize=int(round( float(image_mapsize)*8/3) ) )
        else:
            use_optflow=False
        curent_dset = data_loading.proto_datasets.PoseSequenceDataset(image_wrapper, label_wrapper, key_file, context_length,\
                     image_size = image_size, optical_flow_db_wrapper=optical_flow_db_wrapper)
        dsets.append(curent_dset)
        print("\n")
    if len(dsets)==1:
        dset = dsets[0]
    else:
        dset = torch.utils.data.ConcatDataset(dsets)
    
    dataloader = data_utils.DataLoader(dset, batch_size=1,
                        shuffle=True, num_workers=num_workers)
    print("Dataloader of of length %d" %(len(dataloader)))
    losses = run_epoch(net, dataloader, gpu, loss_func, debug=debug, use_float = use_float)
    print(losses)
    print("RMS error: %f" %(np.mean(losses)))
    plt.plot(np.linspace(0,losses.shape[0]-1,losses.shape[0]),losses)
   # plt.hist(losses)
    plt.show()
import logging
if __name__ == '__main__':
    logging.basicConfig()
    go()    
    