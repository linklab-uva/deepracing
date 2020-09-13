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


ob_normal_dict = {}
ob_dict = {}
ib_normal_dict = {}
ib_dict = {}
def loadTracks(track_id, track_file_dir, device):
    global ob_dict, ob_normal_dict, ib_dict, ib_normal_dict
    trackName = trackNames[track_id]
    print("Loading Track: " + trackName)
    with open(os.path.join(track_file_dir, trackName + "_innerlimit.json"), "r") as f:
        d = json.load(f)
        ib_dict[track_id] = torch.stack([torch.tensor(d["x"], dtype=torch.float64, device=device), torch.tensor(d["z"], dtype=torch.float64, device=device)], dim=1)
        ib_normal_dict[track_id] = F.normalize(torch.stack([torch.tensor(d["x_normal"], dtype=torch.float64, device=device), torch.tensor(d["z_normal"], dtype=torch.float64, device=device)], dim=1), dim=1)
    with open(os.path.join(track_file_dir, trackName + "_outerlimit.json"), "r") as f:
        d = json.load(f)
        ob_dict[track_id] = torch.stack([torch.tensor(d["x"], dtype=torch.float64, device=device), torch.tensor(d["z"], dtype=torch.float64, device=device)], dim=1)
        ob_normal_dict[track_id] = F.normalize(torch.stack([torch.tensor(d["x_normal"], dtype=torch.float64, device=device), torch.tensor(d["z_normal"], dtype=torch.float64, device=device)], dim=1), dim=1)
    print("Loaded Track: " + trackName)



#torch.backends.cudnn.enabled = False
def run_epoch(experiment, network, fix_first_point, optimizer, trainLoader, kinematic_loss, boundary_loss, lossdict, epoch_number, use_label_times = True, imsize=(66,200), timewise_weights=None, debug=False, use_tqdm=True, position_indices=[0,2]):
    global ob_dict, ob_normal_dict, ib_dict, ib_normal_dict
    cum_loss = 0.0
    cum_param_loss = 0.0
    cum_position_loss = 0.0
    cum_velocity_loss = 0.0
    num_samples=0.0
    batch_size = trainLoader.batch_size
    if use_tqdm:
        t = tqdm(enumerate(trainLoader), total=len(trainLoader))
    else:
        t = enumerate(trainLoader)
    network.train()  # This is important to call before training!
    dataloaderlen = len(trainLoader)
    dev = next(network.parameters()).device  # we are only doing single-device training for now, so this works fine.

    _, _, _, _, _, _, sample_session_times,_,_ = trainLoader.dataset[0]
    s_torch = torch.linspace(0.0,1.0,steps=sample_session_times.shape[0],dtype=torch.float64,device=dev).unsqueeze(0).repeat(batch_size,1)
    bezier_order = network.params_per_dimension-1+int(fix_first_point)
    if not debug:
        experiment.set_epoch(epoch_number)
    for (i, (image_torch, key_indices_torch, positions_torch, quats_torch, linear_velocities_torch, angular_velocities_torch, session_times_torch, car_poses, track_ids) ) in t:
        image_torch = image_torch.double().to(device=dev)
        positions_torch = positions_torch.double().to(device=dev)
        session_times_torch = session_times_torch.double().to(device=dev)
        linear_velocities_torch = linear_velocities_torch.double().to(device=dev)
        car_poses = car_poses.double().to(device=dev)
        image_keys = ["image_%d" % (key_indices_torch[j],) for j in range(key_indices_torch.shape[0])]
        
        predictions = network(image_torch)
        if fix_first_point:
            initial_zeros = torch.zeros(image_torch.shape[0],1,2,dtype=torch.float64,device=image_torch.device)
            network_output_reshape = predictions.transpose(1,2)
            predictions_reshape = torch.cat((initial_zeros,network_output_reshape),dim=1)
        else:
            predictions_reshape = predictions.transpose(1,2)
        current_batch_size=session_times_torch.shape[0]
        current_timesteps=session_times_torch.shape[1]
        if use_label_times:
            dt = session_times_torch[:,-1]-session_times_torch[:,0]
            s_torch_cur = (session_times_torch - session_times_torch[:,0,None])/dt[:,None]
        else:
            dt = torch.ones(current_batch_size,dtype=positions_torch.dtype,device=positions_torch.device)
            s_torch_cur = torch.stack([torch.linspace(0.0,1.0,steps=current_timesteps,dtype=positions_torch.dtype,device=positions_torch.device)  for i in range(current_batch_size)], dim=0)

        
        
        gt_points = positions_torch[:,:,position_indices]
        gt_vels = linear_velocities_torch[:,:,position_indices]
        Mpos, controlpoints_fit = deepracing_models.math_utils.bezier.bezierLsqfit(gt_points, bezier_order, t = s_torch_cur)
        fit_points = torch.matmul(Mpos, controlpoints_fit)

        Mvel, fit_vels = deepracing_models.math_utils.bezier.bezierDerivative(controlpoints_fit, t = s_torch_cur, order=1)
        fit_vels_scaled = fit_vels/dt[:,None,None]
        

        pred_points = torch.matmul(Mpos, predictions_reshape)
        
        _, pred_vels = deepracing_models.math_utils.bezier.bezierDerivative(predictions_reshape, t = s_torch_cur, order=1)
        pred_vels_scaled = pred_vels/dt[:,None,None]
        
        use_boundaries = not torch.any(torch.isnan(track_ids))
        if debug:
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
            images_np = np.round(255.0*image_torch[0].detach().cpu().numpy().copy().transpose(0,2,3,1)).astype(np.uint8)
            #image_np_transpose=skimage.util.img_as_ubyte(images_np[-1].transpose(1,2,0))
            ims = []
            for i in range(images_np.shape[0]):
                ims.append([ax1.imshow(images_np[i])])
            ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True, repeat=True)
            # fig2 = plt.figure()
           # ax1.imshow(ani)


            gt_points_np = gt_points[0].detach().cpu().numpy().copy()
            pred_points_np = pred_points[0].detach().cpu().numpy().copy()
            pred_control_points_np = predictions_reshape[0].detach().cpu().numpy().copy()
            #print(gt_points_np)
            fit_points_np = fit_points[0].cpu().numpy().copy()
            fit_control_points_np = controlpoints_fit[0].cpu().numpy().copy()
            


            gt_vels_np = gt_vels[0].cpu().numpy().copy()
            fit_vels_np = fit_vels_scaled[0].cpu().numpy().copy()


            ax2.plot(-gt_points_np[:,0],gt_points_np[:,1],'g+', label="Ground Truth Waypoints")
            ax2.plot(-fit_points_np[:,0],fit_points_np[:,1],'b-', label="Best-fit Bézier Curve")
            ax2.scatter(-fit_control_points_np[1:,0],fit_control_points_np[1:,1],c="b", label="Bézier Curve's Control Points")
            ax2.scatter(-fit_control_points_np[0,0],fit_control_points_np[0,1],c="g", label="This should be (0,0)")
            ax2.plot(-pred_points_np[:,0],pred_points_np[:,1],'r-', label="Predicted Bézier Curve")
            ax2.scatter(-pred_control_points_np[:,0],pred_control_points_np[:,1], c='r', label="Predicted Bézier Curve's Control Points")
           
            velocity_err = kinematic_loss(fit_vels_scaled, gt_vels).item()
            # print("\nMean velocity error: %f\n" % (velocity_err))
            # print(session_times_torch)
            # print(s_torch_cur)
            # print(dt)
            plt.show()

        current_position_loss = kinematic_loss(pred_points, gt_points)
        current_velocity_loss = kinematic_loss(pred_vels_scaled, gt_vels)
        current_param_loss = kinematic_loss(predictions_reshape,controlpoints_fit)

        position_weight = lossdict["position"]
        velocity_weight = lossdict["velocity"]
        param_weight = lossdict["control_point"]
        
       # print(track_ids)
        kinematic_losses = position_weight*current_position_loss + velocity_weight*current_velocity_loss

        if use_boundaries:
            ib = torch.stack([ib_dict[track_ids[i].item()] for i in range(track_ids.shape[0])], dim=0)
            ob = torch.stack([ob_dict[track_ids[i].item()] for i in range(track_ids.shape[0])], dim=0)
            ib_normal = torch.stack([ib_normal_dict[track_ids[i].item()] for i in range(track_ids.shape[0])], dim=0)
            ob_normal = torch.stack([ob_normal_dict[track_ids[i].item()] for i in range(track_ids.shape[0])], dim=0)

            pred_points_aug = torch.stack([pred_points[:,:,0], torch.zeros_like(pred_points[:,:,0]) , pred_points[:,:,1], torch.ones_like(pred_points[:,:,0])], dim=1)
            pred_points_global = torch.matmul(car_poses, pred_points_aug)[:,[0,2],:].transpose(1,2)
            ibloss = boundary_loss(pred_points_global, ib, ib_normal)
            obloss = boundary_loss(pred_points_global, ob, ob_normal)
            loss = kinematic_losses + lossdict["boundary"]["inner_weight"]*ibloss + lossdict["boundary"]["outer_weight"]*obloss
            iblossfloat = float(ibloss.item())
            oblossfloat = float(obloss.item())
        else:
            loss = kinematic_losses
            iblossfloat = 0.0
            oblossfloat = 0.0
        # Backward pass:
        optimizer.zero_grad()
        loss.backward() 
        # Weight and bias updates.
        optimizer.step()
        # logging information
        cum_loss = float(loss.item())
        cum_param_loss = float(current_param_loss.item())
        cum_position_loss = float(current_position_loss.item())
        cum_velocity_loss = float(current_velocity_loss.item())
        num_samples += 1.0
        if not debug:
            experiment.log_metric("position_error", cum_position_loss, step=(epoch_number-1)*dataloaderlen + i)
            experiment.log_metric("velocity_error", cum_velocity_loss, step=(epoch_number-1)*dataloaderlen + i)
            experiment.log_metric("inner_boundary_loss", iblossfloat, step=(epoch_number-1)*dataloaderlen + i)
            experiment.log_metric("outer_boundary_loss", oblossfloat, step=(epoch_number-1)*dataloaderlen + i)
        if use_tqdm:
            t.set_postfix({"position_loss" : cum_position_loss, "velocity_loss" : cum_velocity_loss, "inner_boundary_loss" : iblossfloat, "outer_boundary_loss" : oblossfloat})
def go():
    global ob_dict, ob_normal_dict, ib_dict, ib_normal_dict
    parser = argparse.ArgumentParser(description="Train AdmiralNet Pose Predictor")
    parser.add_argument("dataset_config_file", type=str,  help="Dataset Configuration file to load")
    parser.add_argument("model_config_file", type=str,  help="Model Configuration file to load")
    parser.add_argument("output_directory", type=str,  help="Where to put the resulting model files")

    parser.add_argument("--context_length",  type=int, default=None,  help="Override the context length specified in the config file")
    parser.add_argument("--debug", action="store_true",  help="Display images upon each iteration of the training loop")
    parser.add_argument("--model_load",  type=str, default=None,  help="Load this model file prior to running. usually in conjunction with debug")
    parser.add_argument("--models_to_disk", action="store_true",  help="Save the model files to disk in addition to comet.ml")
    parser.add_argument("--override", action="store_true",  help="Delete output directory and replace with new data")
    parser.add_argument("--tqdm", action="store_true",  help="Display tqdm progress bar on each epoch")
    parser.add_argument("--batch_size", type=int, default=None,  help="Override the order of the batch size specified in the config file")
    parser.add_argument("--gpu", type=int, default=None,  help="Override the GPU index specified in the config file")
    parser.add_argument("--learning_rate", type=float, default=None,  help="Override the learning rate specified in the config file")
    parser.add_argument("--momentum", type=float, default=None,  help="Override the momentum specified in the config file")
    parser.add_argument("--dampening", type=float, default=None,  help="Override the dampening specified in the config file")
    parser.add_argument("--nesterov", action="store_true",  help="Override the nesterov specified in the config file")
    parser.add_argument("--bezier_order", type=int, default=None,  help="Override the order of the bezier curve specified in the config file")
    parser.add_argument("--weighted_loss", action="store_true",  help="Use timewise weights on param loss")
    parser.add_argument("--optimizer", type=str, default="SGD",  help="Optimizer to use")
    parser.add_argument("--velocity_loss", type=float, default=None,  help="Override velocity loss weight in config file")
    parser.add_argument("--position_loss", type=float, default=None,  help="Override position loss weight in config file")
    parser.add_argument("--control_point_loss", type=float, default=None,  help="Override control point loss weight in config file")
    parser.add_argument("--fix_first_point",type=bool,default=False, help="Override fix_first_point in the config file")
    
    args = parser.parse_args()

    dataset_config_file = args.dataset_config_file
    debug = args.debug
    weighted_loss = args.weighted_loss
    model_load = args.model_load
    models_to_disk = args.models_to_disk

    with open(dataset_config_file) as f:
        dataset_config = yaml.load(f, Loader = yaml.SafeLoader)
    config_file = args.model_config_file
    with open(config_file) as f:
        config = yaml.load(f, Loader = yaml.SafeLoader)
    print(dataset_config)
    image_size = dataset_config["image_size"]
    input_channels = config["input_channels"]
    
    if args.context_length is not None:
        context_length = args.context_length
        config["context_length"]  = context_length
    else:
        context_length = config["context_length"]
    if args.bezier_order is not None:
        bezier_order = args.bezier_order
        config["bezier_order"]  = bezier_order
    else:
        bezier_order = config["bezier_order"]
    #num_recurrent_layers = config["num_recurrent_layers"]
    if args.gpu is not None:
        gpu = args.gpu
        config["gpu"]  = gpu
    else:
        gpu = config["gpu"] 
    torch.cuda.set_device(gpu)
    if args.batch_size is not None:
        batch_size = args.batch_size
        config["batch_size"]  = batch_size
    else:
        batch_size = config["batch_size"]
    if args.learning_rate is not None:
        learning_rate = args.learning_rate
        config["learning_rate"] = learning_rate
    else:
        learning_rate = config["learning_rate"]
    if args.momentum is not None:
        momentum = args.momentum
        config["momentum"] = momentum
    else:
        momentum = config["momentum"]
    if args.dampening is not None:
        dampening = args.dampening
        config["dampening"] = dampening
    else:
        dampening = config["dampening"]
    if args.nesterov:
        nesterov = True
        config["nesterov"] = nesterov
    else:
        nesterov = config["nesterov"]
    if args.fix_first_point:
        fix_first_point = True
        config["fix_first_point"] = fix_first_point
    else:
        fix_first_point = config["fix_first_point"]
   
    num_epochs = config["num_epochs"]
    num_workers = config["num_workers"]
    hidden_dim = config["hidden_dimension"]
    use_3dconv = config["use_3dconv"]
    num_recurrent_layers = config.get("num_recurrent_layers",1)
    config["hostname"] = socket.gethostname()
    lossdict = config["loss"]
    track_file_dir = config.get("track_file_dir", os.environ.get("F1_TRACK_DIR", None))
    if track_file_dir is None:
        raise ValueError("Must either set track_file_dir in the training config or set the F1_TRACK_DIR environment variable")
    
    
    print("Using config:\n%s" % (str(config)))
    net = deepracing_models.nn_models.Models.AdmiralNetCurvePredictor( context_length = context_length , input_channels=input_channels, hidden_dim = hidden_dim, num_recurrent_layers=num_recurrent_layers, params_per_dimension=bezier_order + 1 - int(fix_first_point), use_3dconv = use_3dconv) 
    print("net:\n%s" % (str(net)))
    ppd = net.params_per_dimension
    numones = int(ppd/2)
    
    kinematic_loss = deepracing_models.nn_models.LossFunctions.SquaredLpNormLoss()
    boundary_loss = deepracing_models.nn_models.LossFunctions.BoundaryLoss(alpha = lossdict["boundary"]["alpha"], beta=lossdict["boundary"]["beta"])

    print("casting stuff to double")
    net = net.double()
    kinematic_loss = kinematic_loss.double()
    boundary_loss = boundary_loss.double()
    if model_load is not None:
        net.load_state_dict(torch.load(model_load, map_location=torch.device("cpu")))
    if gpu>=0:
        print("moving stuff to GPU")
        device = torch.device("cuda:%d" % gpu)
        net = net.cuda(gpu)
        kinematic_loss = kinematic_loss.cuda(gpu)
        boundary_loss = boundary_loss.cuda(gpu)
    else:
        device = torch.device("cpu")
    optimizer = args.optimizer
    


    config["optimizer"] = optimizer
    if optimizer=="Adam":
        optimizer = optim.Adam(net.parameters(), lr = learning_rate, betas=(0.9, 0.9))
    elif optimizer=="RMSprop":
        optimizer = optim.RMSprop(net.parameters(), lr = learning_rate, momentum = momentum)
    elif optimizer=="ASGD":
        optimizer = optim.ASGD(net.parameters(), lr = learning_rate)
    elif optimizer=="SGD":
        nesterov_ = momentum>0.0 and nesterov
        dampening_=dampening*float(not nesterov_)
        optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum = momentum, dampening=dampening_, nesterov=nesterov_)
    else:
        raise ValueError("Uknown optimizer " + optimizer)

   
    
    
    
        
    if num_workers == 0:
        max_spare_txns = 50
    else:
        max_spare_txns = num_workers

    #image_wrapper = deepracing.backend.ImageFolderWrapper(os.path.dirname(image_db))
    
    dsets=[]
    use_optflow = net.input_channels==5
    dsetfolders = []
    position_indices = dataset_config["position_indices"]
    print("Extracting position indices: %s" %(str(position_indices)))
    alltags = set([])
    dset_output_lengths=[]
    lookahead_indices = dataset_config["lookahead_indices"]   
    use_label_times = dataset_config["use_label_times"]
    use_boundary_loss = dataset_config.get("boundary_loss", False) 
    for dataset in dataset_config["datasets"]:
        print("Parsing database config: %s" %(str(dataset)))
        lateral_dimension = dataset["lateral_dimension"]  
        geometric_variants = dataset.get("geometric_variants", False)   
        gaussian_blur_radius = dataset.get("gaussian_blur_radius", None)
        label_subfolder = dataset.get("label_subfolder", "pose_sequence_labels")
        color_jitter = dataset.get("color_jitter", None)
        key_file = dataset["key_file"]
        dataset_tags = dataset.get("tags", [])
        alltags = alltags.union(set(dataset_tags))
        root_folder = dataset["root_folder"]
        dsetfolders.append(root_folder)
        label_folder = os.path.join(root_folder,label_subfolder)
        image_folder = os.path.join(root_folder,"images")
        key_file = os.path.join(root_folder,key_file)
        label_wrapper = deepracing.backend.PoseSequenceLabelLMDBWrapper()
        label_wrapper.readDatabase(os.path.join(label_folder,"lmdb") )

        image_mapsize = float(np.prod(image_size)*3+12)*float(len(label_wrapper.getKeys()))*1.1
        image_wrapper = deepracing.backend.ImageLMDBWrapper(direct_caching=False)
        image_wrapper.readDatabase( os.path.join(image_folder,"image_lmdb"), mapsize=image_mapsize )
        if use_boundary_loss:
            try:
                with open(os.path.join(label_folder,"config.yaml"),"r") as f:
                    metadata = yaml.load(f, Loader=yaml.SafeLoader)
                    print("Openned metadata: %s" %(str(metadata),))
                    track_id = metadata["track_id"]
                    if not track_id in ib_dict:
                        loadTracks(track_id, track_file_dir, device)
            except Exception as e:
                track_id = np.nan
        else:
            track_id = np.nan


        curent_dset = PD.PoseSequenceDataset(image_wrapper, label_wrapper, key_file, context_length, track_id,\
                     image_size = image_size, lookahead_indices = lookahead_indices, lateral_dimension=lateral_dimension, \
                     geometric_variants = geometric_variants, gaussian_blur=gaussian_blur_radius, color_jitter = color_jitter)
        dsets.append(curent_dset)
        _, _, positions_test, _, _, _, session_times_test ,_ , _ = curent_dset[0]
        dset_output_lengths.append(positions_test.shape[0])
        print("\n")
    if len(dsets)==1:
        dset = dsets[0]
    else:
        dset = torch.utils.data.ConcatDataset(dsets)
    
    dataloader = data_utils.DataLoader(dset, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers, pin_memory=gpu>=0)
    print("Dataloader of of length %d" %(len(dataloader)))

    
    main_dir = args.output_directory
    if debug:
        output_directory = os.path.join(main_dir, "debug")
        os.makedirs(output_directory, exist_ok=True)
    else:
        experiment = comet_ml.Experiment(workspace="electric-turtle", project_name="deepracingbezierpredictor")
        output_directory = os.path.join(main_dir, experiment.get_key())
        if os.path.isdir(output_directory) :
            raise FileExistsError("%s already exists, this should not happen." %(output_directory) )
        os.makedirs(output_directory)
        experiment.log_parameters(config)
        experiment.log_parameters(dataset_config)
        dsetsjson = json.dumps(dataset_config, indent=1)
        experiment.log_parameter("datasets",dsetsjson)
        experiment.log_parameter("dset_output_lengths",dset_output_lengths)
        experiment.log_text(dsetsjson)
        experiment.add_tag("bezierpredictor")
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
        #def run_epoch(experiment, network, fix_first_point, optimizer, trainLoader, kinematic_loss, boundary_loss, lossdict, epoch_number, 
    if debug:
        run_epoch(None, net, fix_first_point , optimizer, dataloader, kinematic_loss, boundary_loss, lossdict, 1, use_label_times=use_label_times, debug=True, use_tqdm=args.tqdm, position_indices=position_indices)
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
                    run_epoch(experiment, net, fix_first_point , optimizer, dataloader, kinematic_loss, boundary_loss, lossdict, postfix, use_label_times=use_label_times, debug=False, use_tqdm=args.tqdm, position_indices=position_indices)
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
    