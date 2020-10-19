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

#torch.backends.cudnn.enabled = False
def run_epoch(experiment, network, optimizer, dataloader, ego_agent_loss, config, use_tqdm = False, debug=False):
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
    loss_weights = config["loss_weights"]
    bezier_order = network.bezier_order

    for (i, scandict) in t:
        scans = scandict["scans"].double().to(device=dev)
        ego_current_pose = scandict["ego_current_pose"].double().to(device=dev)
        ego_positions = scandict["ego_positions"].double().to(device=dev)
        ego_velocities = scandict["ego_velocities"].double().to(device=dev)
        session_times = scandict["session_times"].double().to(device=dev)
        raceline = scandict["raceline"].double().to(device=dev)
        batch_size = scans.shape[0]
        
        
    
        predictions = network(scans)

        # dt = session_times[:,-1]-session_times[:,0]
        # s_torch_cur = (session_times - session_times[:,0,None])/dt[:,None]
        s_torch_cur = torch.linspace(0.0, 1.0, steps=session_times.shape[1], dtype=torch.float64, device=dev).expand(batch_size,-1)
        Mpos = deepracing_models.math_utils.bezierM(s_torch_cur, bezier_order)
        # print(predictions.shape)
        # print(s_torch_cur.shape)
        # print(Mpos.shape)
        pred_points = torch.matmul(Mpos, predictions)

        # Mvel, pred_vel_s = deepracing_models.math_utils.bezier.bezierDerivative(predictions, t = s_torch_cur, order=1)
        # pred_vel_t = pred_vel_s/dt[:,None,None]
      #  loss = ego_agent_loss(pred_points, ego_positions)
        loss = ego_agent_loss(pred_points, raceline)

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        
        
        if debug and False:
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
            images_np = np.round(255.0*input_images[0].detach().cpu().numpy().copy().transpose(0,2,3,1)).astype(np.uint8)
            #image_np_transpose=skimage.util.img_as_ubyte(images_np[-1].transpose(1,2,0))
            # oap = other_agent_positions[other_agent_positions==other_agent_positions].view(1,-1,60,2)
            # print(oap)
            ims = []
            for i in range(images_np.shape[0]):
                ims.append([ax1.imshow(images_np[i])])
            ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True, repeat=True)


            _, controlpoints_fit = deepracing_models.math_utils.bezier.bezierLsqfit(ego_positions, bezier_order, M = Mpos)
            fit_points = torch.matmul(Mpos, controlpoints_fit)

            gt_points_np = ego_positions[0].detach().cpu().numpy().copy()
            pred_points_np = pred_points[0].detach().cpu().numpy().copy()
            pred_control_points_np = predictions[0].detach().cpu().numpy().copy()
            fit_points_np = fit_points[0].cpu().numpy().copy()
            fit_control_points_np = controlpoints_fit[0].cpu().numpy().copy()
            
            ymin = np.min(np.hstack([gt_points_np[:,1], fit_points_np[:,1], pred_points_np[:,1] ]))-0.05
            ymax = np.max(np.hstack([gt_points_np[:,1], fit_points_np[:,1], pred_points_np[:,1] ]))+0.05
            xmin = np.min(np.hstack([gt_points_np[:,0], fit_points_np[:,0] ])) - 0.05
            xmax = np.max(np.hstack([gt_points_np[:,0], fit_points_np[:,0] ]))
            ax2.set_xlim(ymax,ymin)
            ax2.set_ylim(xmin,xmax)
            ax2.plot(gt_points_np[:,1],gt_points_np[:,0],'g+', label="Ground Truth Waypoints")
            ax2.plot(fit_points_np[:,1],fit_points_np[:,0],'b-', label="Best-fit Bézier Curve")
            ax2.plot(pred_points_np[:,1],pred_points_np[:,0],'r-', label="Predicted Bézier Curve")
            plt.legend()
            plt.show()

        if not debug:
            experiment.log_metric("current_position_loss", loss)
        if use_tqdm:
            t.set_postfix({"current_position_loss" : float(loss.item())})
def go():
    parser = argparse.ArgumentParser(description="Train AdmiralNet Pose Predictor")
    parser.add_argument("dataset_config_file", type=str,  help="Dataset Configuration file to load")
    parser.add_argument("model_config_file", type=str,  help="Model Configuration file to load")
    parser.add_argument("output_directory", type=str,  help="Where to put the resulting model files")

    parser.add_argument("--debug", action="store_true",  help="Display images upon each iteration of the training loop")
    parser.add_argument("--model_load",  type=str, default=None,  help="Load this model file prior to running. usually in conjunction with debug")
    parser.add_argument("--models_to_disk", action="store_true",  help="Save the model files to disk in addition to comet.ml")
    parser.add_argument("--tqdm", action="store_true",  help="Display tqdm progress bar on each epoch")
    parser.add_argument("--gpu", type=int, default=None,  help="Override the GPU index specified in the config file")

    
    args = parser.parse_args()

    dataset_config_file = args.dataset_config_file
    debug = args.debug
    model_load = args.model_load
    models_to_disk = args.models_to_disk
    use_tqdm = args.tqdm

    with open(dataset_config_file) as f:
        dataset_config = yaml.load(f, Loader = yaml.SafeLoader)
    config_file = args.model_config_file
    with open(config_file) as f:
        config = yaml.load(f, Loader = yaml.SafeLoader)
    print(dataset_config)
    
    context_length = config["context_length"]
    bezier_order = config["bezier_order"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    momentum = config["momentum"]
    dampening = config["dampening"]
    nesterov = config["nesterov"]
    project_name = config["project_name"]
   
    if args.gpu is not None:
        gpu = args.gpu
        config["gpu"]  = gpu
    else:
        gpu = config["gpu"] 
    torch.cuda.set_device(gpu)

    num_epochs = config["num_epochs"]
    num_workers = config["num_workers"]
    hidden_dim = config["hidden_dimension"]
    input_features = config["input_features"]
    num_recurrent_layers = config.get("num_recurrent_layers",1)
    config["hostname"] = socket.gethostname()

    
    print("Using config:\n%s" % (str(config)))
    net = deepracing_models.nn_models.Models.LinearRecursionCurvePredictor(input_features, hidden_dimension=hidden_dim, bezier_order=bezier_order)
    print("net:\n%s" % (str(net)))
    ego_agent_loss = deepracing_models.nn_models.LossFunctions.SquaredLpNormLoss()
    print("casting stuff to double")
    net = net.double()
    ego_agent_loss = ego_agent_loss.double()

    if model_load is not None:
        net.load_state_dict(torch.load(model_load, map_location=torch.device("cpu")))
    if gpu>=0:
        print("moving stuff to GPU")
        device = torch.device("cuda:%d" % gpu)
        net = net.cuda(gpu)
        ego_agent_loss = ego_agent_loss.cuda(gpu)
    else:
        device = torch.device("cpu")
    optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum = momentum, dampening=dampening, nesterov=nesterov)
    

    #image_wrapper = deepracing.backend.ImageFolderWrapper(os.path.dirname(image_db))
    
    dsets=[]
    dsetfolders = []
    alltags = set(dataset_config.get("tags",[]))
    dset_output_lengths=[]
    return_other_agents = bool(dataset_config.get("other_agents",False))
    for dataset in dataset_config["datasets"]:
        dlocal : dict = {k: dataset_config[k] for k in dataset_config.keys()  if (not (k in ["datasets"]))}
        dlocal.update(dataset)
        print("Parsing database config: %s" %(str(dlocal)))
        root_folder = dlocal["root_folder"]
        position_indices = dlocal["position_indices"]
        dataset_tags = dlocal.get("tags", [])
        alltags = alltags.union(set(dataset_tags))

        dsetfolders.append(root_folder)
        scan_folder = os.path.join(root_folder,"laser_scans")
        label_folder = os.path.join(root_folder,"laser_scan_labels")
        key_file = os.path.join(label_folder,"goodkeys.txt")
        with open(key_file,"r") as f:
            keys = [l.replace("\n","") for l in f.readlines()]
            keys = [k for k in keys if not k==""]
            numkeys = len(keys)
        scan_wrapper = deepracing.backend.LaserScanLMDBWrapper()
        scan_wrapper.openDatabase( os.path.join(scan_folder,"lmdb"), mapsize=38000*numkeys )

        label_wrapper = deepracing.backend.MultiAgentLabelLMDBWrapper()
        label_wrapper.openDatabase(os.path.join(label_folder,"lmdb"), mapsize=17000*numkeys )


        
        current_dset = PD.LaserScanDataset(scan_wrapper, label_wrapper, keys, context_length, position_indices, return_other_agents=return_other_agents)
        dsets.append(current_dset)
        
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
        #def run_epoch(experiment, net, optimizer, dataloader, raceline_loss, other_agent_loss, config)
    if debug:
        run_epoch(experiment, net, optimizer, dataloader, ego_agent_loss, config, debug=True, use_tqdm=True)
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
                    run_epoch(experiment, net, optimizer, dataloader, ego_agent_loss, config, use_tqdm=use_tqdm)
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
    