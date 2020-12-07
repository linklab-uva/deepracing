import comet_ml
import torch
import torch.nn as NN
import torch.nn.functional as F
import torch.utils.data as data_utils
import deepracing_models.data_loading.proto_datasets as PD
from tqdm import tqdm as tqdm
import deepracing_models.nn_models.LossFunctions as loss_functions
import deepracing_models.nn_models.Models
import numpy as np
import torch.optim as optim
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
import deepracing
from deepracing import trackNames
import deepracing.backend
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import deepracing_models.math_utils.bezier
import socket
import json
from comet_ml.api import API, APIExperiment
import cv2
import torchvision, torchvision.transforms as T
from deepracing_models.data_loading.image_transforms import GaussianBlur
from deepracing.raceline_utils import loadBoundary
from deepracing import searchForFile
import deepracing.path_utils.geometric as geometric

#torch.backends.cudnn.enabled = False
def run_epoch(experiment, network, optimizer, dataloader, loss_dict, config, use_tqdm = False, debug=False, plot=False):
    cum_loss = 0.0
    cum_param_loss = 0.0
    cum_position_loss = 0.0
    cum_velocity_loss = 0.0
    num_samples=0.0
    if use_tqdm:
        t = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        t = enumerate(dataloader)
    network.train()  # This is important to call before training!
    dataloaderlen = len(dataloader)
    
    dev = next(network.parameters()).device  # we are only doing single-device training for now, so this works fine.
    dtype = next(network.parameters()).dtype # we are only doing single-device training for now, so this works fine.
    fix_first_point = config["fix_first_point"]
    ego_agent_loss = loss_dict["position"]
    target = config["target"]

    #_, _, _, _, _, _, sample_session_times,_,_ = dataloader.dataset[0]
    bezier_order = network.bezier_order
   # Mpos = deepracing_models.math_utils.bezierM(torch.linspace(0.0,1.5,120).unsqueeze(0).repeat(64,1).double().to(device=dev), bezier_order)

    for (i, imagedict) in t:
        track_names = imagedict["track"]
        input_images = imagedict["images"].type(dtype).to(device=dev)
        session_times = imagedict["session_times"].type(dtype).to(device=dev)
        image_timestamps = imagedict["image_timestamps"].type(dtype).to(device=dev)
        targets = imagedict[target].type(dtype).to(device=dev)

        
        batch_size = input_images.shape[0]
        context_length = input_images.shape[1]

     #   print(image_timestamps)
        dt_images = image_timestamps[:,-1]-image_timestamps[:,0]
        s_images = (image_timestamps - image_timestamps[:,0,None])/dt_images[:,None]

     #   print(s_images)
        images_flat = input_images.view(batch_size, context_length, -1)
     #   print(images_flat.shape)
        M_images, image_curves_flat = deepracing_models.math_utils.bezierLsqfit(images_flat, bezier_order, t=s_images)
        image_curves = torch.clamp(image_curves_flat.view(batch_size, bezier_order+1, network.input_channels, network.input_rows, network.input_columns), 0.0, 1.0)
        
        predicted_curves = network(image_curves)
        # if fix_first_point:
        #     initial_zeros = torch.zeros(batch_size,1,2,dtype=dtype,device=dev)
        #     predicted_curves = torch.cat((initial_zeros,predictions),dim=1)
        # else:
        #     predicted_curves = predictions
        # current_batch_size=image_keys.shape[0]
        # current_timesteps=image_keys.shape[1]
        # if use_label_times:
        #     dt = session_times_torch[:,-1]-session_times_torch[:,0]
        #     s_torch_cur = (session_times_torch - session_times_torch[:,0,None])/dt[:,None]
        # else:
        #     dt = torch.ones(current_batch_size,dtype=positions_torch.dtype,device=positions_torch.device)
        #     s_torch_cur = torch.stack([torch.linspace(0.0,1.0,steps=current_timesteps,dtype=positions_torch.dtype,device=positions_torch.device)  for i in range(current_batch_size)], dim=0)

        
        dt = session_times[:,-1]-session_times[:,0]
        s_torch_cur = (session_times - session_times[:,0,None])/dt[:,None]
        # gt_vels = linear_velocities_torch[:,:,position_indices]
        Mpos = deepracing_models.math_utils.bezierM(s_torch_cur, bezier_order)
        pred_points = torch.matmul(Mpos, predicted_curves)


        
        if debug and plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
            # images_np = np.round(255.0*input_images[0].detach().cpu().numpy().copy().transpose(0,2,3,1)).astype(np.uint8)
            images_np = np.round(255.0*image_curves[0].detach().cpu().numpy().copy().transpose(0,2,3,1)).astype(np.uint8)
            image_curves
            #image_np_transpose=skimage.util.img_as_ubyte(images_np[-1].transpose(1,2,0))
            # oap = other_agent_positions[other_agent_positions==other_agent_positions].view(1,-1,60,2)
            # print(oap)
            ims = []
            for i in range(images_np.shape[0]):
                ims.append([ax1.imshow(images_np[i])])
            ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True, repeat=True)


            _, controlpoints_fit = deepracing_models.math_utils.bezier.bezierLsqfit(targets, bezier_order, M = Mpos)
            fit_points = torch.matmul(Mpos, controlpoints_fit)

            # gt_points_np = ego_positions[0].detach().cpu().numpy().copy()
            gt_points_np = targets[0].detach().cpu().numpy().copy()
            pred_points_np = pred_points[0].detach().cpu().numpy().copy()
            pred_control_points_np = predicted_curves[0].detach().cpu().numpy().copy()
            fit_points_np = fit_points[0].cpu().numpy().copy()
            fit_control_points_np = controlpoints_fit[0].cpu().numpy().copy()
            
            ymin = np.min(np.hstack([gt_points_np[:,1], pred_points_np[:,1] ])) - 2.5
            ymax = np.max(np.hstack([gt_points_np[:,1], pred_points_np[:,1] ])) + 2.5
            xmin = np.min(np.hstack([gt_points_np[:,0], fit_points_np[:,0] ])) -  2.5
            xmax = np.max(np.hstack([gt_points_np[:,0], fit_points_np[:,0] ]))
            ax2.set_xlim(xmax,xmin)
            ax2.set_ylim(ymin,ymax)
            ax2.plot(gt_points_np[:,0],gt_points_np[:,1],'g+', label="Ground Truth Waypoints")
            ax2.plot(pred_points_np[:,0],pred_points_np[:,1],'r-', label="Predicted Bézier Curve")
            # ax2.plot(fit_points_np[:,0],fit_points_np[:,1],'b-', label="Best-fit Bézier Curve")
            #ax2.scatter(fit_control_points_np[1:,0],fit_control_points_np[1:,1],c="b", label="Bézier Curve's Control Points")
       #     ax2.plot(pred_points_np[:,1],pred_points_np[:,0],'r-', label="Predicted Bézier Curve")
          #  ax2.scatter(pred_control_points_np[:,1],pred_control_points_np[:,0], c='r', label="Predicted Bézier Curve's Control Points")
            plt.legend()
            plt.show()

        loss = ego_agent_loss(pred_points, targets)

        optimizer.zero_grad()
        loss.backward() 
        # Weight and bias updates.
        optimizer.step()
        # logging information
        loss_float = float(loss.item())
        if use_tqdm:
            t.set_postfix({"loss" : loss_float})
def go():
    parser = argparse.ArgumentParser(description="Train AdmiralNet Pose Predictor")
    parser.add_argument("dataset_config_file", type=str,  help="Dataset Configuration file to load")
    parser.add_argument("model_config_file", type=str,  help="Model Configuration file to load")
    parser.add_argument("output_directory", type=str,  help="Where to put the resulting model files")

    parser.add_argument("--debug", action="store_true",  help="Don't actually push to comet, just testing")
    parser.add_argument("--plot", action="store_true",  help="Plot images upon each iteration of the training loop")
    parser.add_argument("--model_load",  type=str, default=None,  help="Load this model file prior to running. usually in conjunction with debug")
    parser.add_argument("--models_to_disk", action="store_true",  help="Save the model files to disk in addition to comet.ml")
    parser.add_argument("--tqdm", action="store_true",  help="Display tqdm progress bar on each epoch")
    parser.add_argument("--gpu", type=int, default=None,  help="Override the GPU index specified in the config file")

    
    args = parser.parse_args()

    dataset_config_file = args.dataset_config_file
    debug = args.debug
    plot = args.plot
    model_load = args.model_load
    models_to_disk = args.models_to_disk
    use_tqdm = args.tqdm

    with open(dataset_config_file) as f:
        dataset_config = yaml.load(f, Loader = yaml.SafeLoader)
    config_file = args.model_config_file
    with open(config_file) as f:
        config = yaml.load(f, Loader = yaml.SafeLoader)
    print(dataset_config)
    image_size = dataset_config["image_size"]
    input_channels = config["input_channels"]
    
    context_length = config["context_length"]
    bezier_order = config["bezier_order"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    momentum = config["momentum"]
    project_name = config["project_name"]
    fix_first_point = config["fix_first_point"]
    conv_activation = config["conv_activation"]
    linear_activation = config["linear_activation"]

   
    if args.gpu is not None:
        gpu = args.gpu
        config["gpu"]  = gpu
    else:
        gpu = config["gpu"] 
    torch.cuda.set_device(gpu)

    num_epochs = config["num_epochs"]
    num_workers = config["num_workers"]
    hidden_dim = config["hidden_dimension"]
    use_3dconv = config["use_3dconv"]
    use_float = config["use_float"]
    num_recurrent_layers = config.get("num_recurrent_layers",1)
    config["hostname"] = socket.gethostname()

    print("Using config:\n%s" % (str(config)))
    
    ego_agent_loss = deepracing_models.nn_models.LossFunctions.SquaredLpNormLoss()

    

    #image_wrapper = deepracing.backend.ImageFolderWrapper(os.path.dirname(image_db))
    
    dsets=[]
    dsetfolders = []
    alltags = set(dataset_config.get("tags",[]))
    dset_output_lengths=[]
    track_name = dataset_config["track_name"]
    image_size = dataset_config["image_size"]
    
    for dataset in dataset_config["datasets"]:
        dlocal : dict = {k: dataset_config[k] for k in dataset_config.keys()  if (not (k in ["datasets"]))}
        dlocal.update(dataset)
        print("Parsing database config: %s" %(str(dlocal)))
        key_file = dlocal["key_file"]
        root_folder = dlocal["root_folder"]
        position_indices = dlocal["position_indices"]
        label_subfolder = dlocal["label_subfolder"]
        track_name =  dlocal["track_name"]
        dataset_tags = dlocal.get("tags", [])
        alltags = alltags.union(set(dataset_tags))

        dsetfolders.append(root_folder)
        label_folder = os.path.join(root_folder,label_subfolder)
        with open(os.path.join(label_folder,"config.yaml"), "r") as f:
            dataset.update(yaml.load(f, Loader=yaml.SafeLoader))

        image_folder = os.path.join(root_folder,"images")
        key_file = os.path.join(root_folder,key_file)
        label_wrapper = deepracing.backend.MultiAgentLabelLMDBWrapper()
        label_wrapper.openDatabase(os.path.join(label_folder,"lmdb") )


        image_lmdb_folder = os.path.join(image_folder,"image_lmdb")
        with open(os.path.join(image_lmdb_folder,"config.yaml"),"r") as f:
            dataset.update(yaml.load(f, Loader=yaml.SafeLoader))


        image_mapsize = float(np.prod(image_size)*3+12)*float(len(label_wrapper.getKeys()))*1.1
        image_wrapper = deepracing.backend.ImageLMDBWrapper()
        image_wrapper.readDatabase( image_lmdb_folder , mapsize=image_mapsize )

        extra_transforms = []
        color_jitters = dlocal.get("color_jitters", None) 
        if color_jitters is not None:
            extra_transforms+=[T.ColorJitter(brightness=[cj, cj]) for cj in color_jitters]
            
        blur = dlocal.get("blur", None)   
        if blur is not None:
            extra_transforms.append(GaussianBlur(blur))
        
        current_dset = PD.MultiAgentDataset(image_wrapper, label_wrapper, key_file, context_length, image_size, position_indices, track_name, extra_transforms=extra_transforms, return_other_agents=False)
        dsets.append(current_dset)
        
        print("\n")
    if len(dsets)==1:
        dset = dsets[0]
    else:
        dset = torch.utils.data.ConcatDataset(dsets)
    
    dataloader = data_utils.DataLoader(dset, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers, pin_memory=gpu>=0)
                        
    
    net = deepracing_models.nn_models.Models.ConvolutionalCurveToCurvePredictor(input_channels=input_channels, input_rows=image_size[0], input_columns=image_size[1], output_dim=2, bezier_order=bezier_order, conv_activation=conv_activation, linear_activation=linear_activation )

    print("net:\n%s" % (str(net)))
    if use_float:
        net = net.float()
    else:
        net = net.double()
    if gpu>=0:
        print("moving stuff to GPU")
        device = torch.device("cuda:%d" % gpu)
        net = net.cuda(gpu)
        ego_agent_loss = ego_agent_loss.cuda(gpu)
    else:
        device = torch.device("cpu")
    optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum = momentum)

    loss_dict = {"position":ego_agent_loss}#, "other_agents": other_agent_loss, "inner_boundary": ibloss, "outer_boundary": obloss}
    dtype = next(net.parameters()).dtype
    ego_agent_loss = ego_agent_loss.type(dtype)


    print("Dataloader of of length %d" %(len(dataloader)))
    if debug:
        print("Using datasets:\n%s", (str(dataset_config)))
    
    main_dir = args.output_directory
    if debug:
        output_directory = os.path.join(main_dir, "debug")
        os.makedirs(output_directory, exist_ok=True)
        experiment = None
    else:
        experiment = comet_ml.Experiment(workspace="electric-turtle", project_name=project_name)
        output_directory = os.path.join(main_dir, experiment.get_key())
        if os.path.isdir(output_directory) :
            raise FileExistsError("%s already exists, this should not happen." %(output_directory) )
        os.makedirs(output_directory)
        experiment.log_parameters(config)
        experiment.log_parameters(dataset_config)
        dsetsjson = json.dumps(dataset_config, indent=1)
        experiment.log_parameter("datasets",dsetsjson)
        experiment.log_text(dsetsjson)
        if len(alltags)>0:
            experiment.add_tags(list(alltags))
        experiment_config = {"experiment_key": experiment.get_key()}
        yaml.dump(experiment_config, stream=open(os.path.join(output_directory,"experiment_config.yaml"),"w"), Dumper=yaml.SafeDumper)
        yaml.dump(dataset_config, stream=open(os.path.join(output_directory,"dataset_config.yaml"), "w"), Dumper = yaml.SafeDumper)
        yaml.dump(config, stream=open(os.path.join(output_directory,"model_config.yaml"), "w"), Dumper = yaml.SafeDumper)
        experiment.log_asset(os.path.join(output_directory,"dataset_config.yaml"),file_name="datasets.yaml")
        experiment.log_asset(os.path.join(output_directory,"experiment_config.yaml"),file_name="experiment_config.yaml")
        experiment.log_asset(os.path.join(output_directory,"model_config.yaml"),file_name="model_config.yaml")
        i = 0
        #def run_epoch(experiment, net, optimizer, dataloader, raceline_loss, other_agent_loss, config)
    if debug:
        run_epoch(experiment, net, optimizer, dataloader, loss_dict, config, debug=True, use_tqdm=True, plot=plot)
    else:
        netpostfix = "epoch_%d_params.pt"
        optimizerpostfix = "epoch_%d_optimizer.pt"
        with experiment.train():
            while i < num_epochs:
                time.sleep(2.0)
                postfix = i + 1
                if models_to_disk:
                    modelfile = netpostfix % (postfix-1)
                    optimizerfile = optimizerpostfix % (postfix-1)
                else:
                    modelfile = "params.pt"
                    optimizerfile = "optimizer.pt"
                print("Running Epoch Number %d" %(postfix))
                #dset.clearReaders()
                try:
                    tick = time.time()
                    run_epoch(experiment, net, optimizer, dataloader, loss_dict, config, use_tqdm=use_tqdm)
                    tock = time.time()
                    print("Finished epoch %d in %f seconds." % ( postfix , tock-tick ) )
                    experiment.log_epoch_end(postfix)
                except FileExistsError as e:
                    raise e
                except Exception as e:
                    print("Restarting epoch %d because %s"%(postfix, str(e)))
                    modelin = os.path.join(output_directory, modelfile)
                    optimizerin = os.path.join(output_directory,optimizerfile)
                    net.load_state_dict(torch.load(modelin))
                    optimizer.load_state_dict(torch.load(optimizerin))
                    continue

                modelout = os.path.join(output_directory,modelfile)
                with open(modelout,'wb') as f:
                    torch.save(net.state_dict(), f)
                with open(modelout,'rb') as f:
                    experiment.log_asset(f,file_name=netpostfix %(postfix,) )

                optimizerout = os.path.join(output_directory, optimizerfile)
                with open(optimizerout,'wb') as f:
                    torch.save(optimizer.state_dict(), f)
                with open(optimizerout,'rb') as f:
                    experiment.log_asset(f,file_name=optimizerpostfix %(postfix,) )
                i = i + 1
import logging
if __name__ == '__main__':
    logging.basicConfig()
    go()    
    