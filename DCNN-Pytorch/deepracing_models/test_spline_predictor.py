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
def run_epoch(network, testLoader, gpu, loss_func, imsize=(66,200), debug=False,  use_float=True):
    lossfloat = 0.0
    batch_size = testLoader.batch_size
    num_samples=0.0
    splK = 3
    numpoints = 11
    t : tqdm = tqdm(enumerate(testLoader), total=len(testLoader))
    network.eval()  # This is important to call before testing!
    for (i, (image_torch, opt_flow_torch, _, _, _, _, _, pos_spline_params, vel_spline_params, knots_torch) ) in t:
        if network.input_channels==5:
            image_torch = torch.cat((image_torch,opt_flow_torch),axis=2)
        if use_float:
            image_torch = image_torch.float()
            pos_spline_params = pos_spline_params.float()
        else:
            image_torch = image_torch.double()
            pos_spline_params = pos_spline_params.double()

        if gpu>=0:
            image_torch = image_torch.cuda(gpu)
            pos_spline_params = pos_spline_params.cuda(gpu)
 
        predictions = network(image_torch)
        loss = loss_func(predictions, pos_spline_params)
        lossfloat+=loss.item()
        num_samples += 1.0
        if debug:
            images_np = image_torch[0].cpu().numpy().copy()
            print(images_np.shape)
            num_images = images_np.shape[0]
            print(num_images)
            images_np_transpose = np.zeros((num_images, images_np.shape[2], images_np.shape[3], 3), dtype=np.uint8)
            ims = []
            fig2 = plt.figure()
            for i in range(num_images):
                imnp = None
                images_np_transpose[i]=np.round(255.0*(images_np[i][0:3]).transpose(1,2,0)).astype(np.uint8)
                im = plt.imshow(images_np_transpose[i], animated=True)
                ims.append([im])
            ani = animation.ArtistAnimation(fig2, ims, interval=50, blit=True, repeat_delay=0)
            tsamp = np.linspace(0,1,60)
            pred_params_np = predictions[0].detach().cpu().numpy().copy().transpose()
            gt_params_np = pos_spline_params[0].detach().cpu().numpy().copy().transpose()
            knots = knots_torch.detach().cpu().numpy()[0].copy()
            pred_spline = scipy.interpolate.BSpline(knots,pred_params_np,splK)
            gt_spline = scipy.interpolate.BSpline(knots,gt_params_np,splK)
            gt_spline_deriv = gt_spline.derivative()
            pred_spline_deriv = pred_spline.derivative()

            gt_positions = gt_spline(tsamp)
            pred_positions = pred_spline(tsamp)
            gt_vels = gt_spline_deriv(tsamp)
            pred_vels = pred_spline_deriv(tsamp)
            gt_speeds = la.norm(gt_vels,axis=1)
            pred_speeds = la.norm(pred_vels,axis=1)

            fig = plt.figure()
            plt.plot(gt_positions[:,0], gt_positions[:,1], 'ro')
            plt.plot(pred_positions[:,0], pred_positions[:,1], 'bo')
            minx = np.min(gt_positions[:,0])
            maxx = np.max(gt_positions[:,0])
            dx = maxx - minx
            miny = np.min(gt_positions[:,1])
            maxy = np.max(gt_positions[:,1])
            dy = maxy - miny
            plt.xlim(minx-0.25*dx,maxx+0.25*dx)
            plt.ylim(miny-0.25*dy,maxy+0.25*dy)
            figvel = plt.figure()
            plt.xlim(0.0,1.0)
            plt.ylim(0.0,125)
            plt.plot(tsamp, gt_speeds, 'ro')
            plt.plot(tsamp, pred_speeds, 'bo')
            plt.show()
        t.set_postfix({"posloss":lossfloat/num_samples})
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
    sequence_length = config["sequence_length"]
    context_length = config["context_length"]
    gpu = args.gpu
    loss_weights = config["loss_weights"]
    temporal_conv_feature_factor = config["temporal_conv_feature_factor"]
    debug = args.debug
    use_float = config["use_float"]
    learnable_initial_state = config.get("learnable_initial_state",True)
    print("Using config:\n%s" % (str(config)))
    net = nn_models.Models.AdmiralNetCurvePredictor(input_channels=input_channels) 
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
        curent_dset = data_loading.proto_datasets.PoseSequenceDataset(image_wrapper, label_wrapper, key_file, context_length, sequence_length,\
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
    run_epoch(net, dataloader, gpu, loss_func, debug=debug, use_float = use_float)
import logging
if __name__ == '__main__':
    logging.basicConfig()
    go()    
    