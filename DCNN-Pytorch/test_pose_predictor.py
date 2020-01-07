import torch
import torch.nn as NN
import torch.utils.data as data_utils
import data_loading.proto_datasets
from tqdm import tqdm as tqdm
import deepracing_models.nn_models.LossFunctions as loss_functions
import deepracing_models.nn_models.Models
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
import deepracing_models.math_utils.bezier
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
    dtarr=np.zeros(len(testLoader.dataset))
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
        dt = session_times_torch[:,-1]-session_times_torch[:,0]
        s_torch = (session_times_torch - session_times_torch[:,0,None])/dt[:,None]
        fitpoints = positions_torch[:,:,[0,2]]      
        loss = torch.mean( torch.sqrt(loss_func(predictions,fitpoints)),dim=1 )
       # print(loss.shape)

        currloss = float(torch.sum(loss).item())
        lossfloat+=currloss
        if testLoader.batch_size==1:
            losses[i]=currloss
            gterrors[i]=torch.mean(torch.norm(fitpoints-gtevalpoints,p=2,dim=2)).item()
            gtvelerrors[i]=torch.mean(torch.norm(fitvels-gt_fit_vels/dt[:,None,None],p=2,dim=2)).item()
            dtarr[i] = dt.item()
        else:
            istart = i*testLoader.batch_size
            iend = istart+s_torch.shape[0]
            losses[istart:iend]=loss.detach().cpu().numpy()
            gterrors[istart:iend]=torch.mean(torch.norm(fitpoints-gtevalpoints,p=2,dim=2),dim=1).detach().cpu().numpy()
            gtvelerrors[istart:iend]=torch.mean(torch.norm(fitvels-gt_fit_vels/dt[:,None,None],p=2,dim=2),dim=1).detach().cpu().numpy()
        num_samples += testLoader.batch_size
        if debug and i%30==0:
            print("Current loss: %s" %(str(loss)))
            predevalpoints = torch.matmul(Mfit, predictions_reshape)
            xgtnp = fitpoints[0,:,0].cpu().detach().numpy()
            ygtnp = fitpoints[0,:,1].cpu().detach().numpy()
            xprednp = predevalpoints[0,:,0].cpu().detach().numpy()
            zprednp = predevalpoints[0,:,1].cpu().detach().numpy()
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot([],[],'g',label="BSpline Samples")
            ax.scatter(-xgtnp,ygtnp, facecolors='none', edgecolors='g')

            predcolor = 'r'
            ax.plot([],[],predcolor,label="Predictions")
            ax.scatter(-xprednp[:,0],zprednp[:,1], facecolors='none', edgecolors=predcolor)
            ax.set_title("Local Ground-Truth Trajectory")

            ax.legend()

            skipn = 1

            plt.show()
        t.set_postfix({"posloss":lossfloat/num_samples})
    return losses, gterrors, gtvelerrors, dtarr

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
    # with open(dataset_config_file,'r') as f:
    #     datasets_config = yaml.load(f, Loader = yaml.SafeLoader)
    # with open(comet_config_file,'r') as f:
    #     comet_config = yaml.load(f, Loader = yaml.SafeLoader)

    image_size = np.array((66,200))
    hidden_dimension = config["hidden_dimension"]
    input_channels = config["input_channels"]
    context_length = config["context_length"]
    sequence_length = config["sequence_length"]
    gpu = args.gpu
    temporal_conv_feature_factor = config["temporal_conv_feature_factor"]
    debug = args.debug
    use_float = False
    learnable_initial_state = config.get("learnable_initial_state",True)
    print("Using config:\n%s" % (str(config)))
    net = nn_models.Models.AdmiralNetKinematicPredictor(context_length= context_length, sequence_length=sequence_length, input_channels=input_channels) 
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
    losses, gterrors, gtvelerrors, dtarr = run_epoch(net, dataloader, gpu, loss_func, debug=debug, use_float = use_float)
    #distances = np.sqrt(losses)
    distance_sort = np.argsort(losses)
    distances = losses.copy()[distance_sort]
    distance_indices = np.linspace(0,distances.shape[0]-1,distances.shape[0]).astype(np.float32)

    distance_sort_descending = np.flip(distance_sort.copy())
    distances_sorted_descending = losses.copy()[distance_sort_descending]
    meandist = np.mean(distances)
    stddist = np.std(distances)
    print(dtarr)
    print("Mean dt: %f" % ( np.mean(dtarr) ) )
    print("STD dt: %f" % ( np.std(dtarr) ) )
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
   
import logging
if __name__ == '__main__':
    logging.basicConfig()
    go()    
    