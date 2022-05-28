import comet_ml
import torch
import torch.utils.data as data_utils
import deepracing_models.data_loading.file_datasets as FD
from tqdm import tqdm as tqdm
import deepracing_models.nn_models.LossFunctions as loss_functions
import deepracing_models.nn_models.Models
import numpy as np
import torch.optim as optim
import os
import yaml
import matplotlib.figure, matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import deepracing_models.math_utils.bezier
import socket

#torch.backends.cudnn.enabled = False
def run_epoch(experiment, network, optimizer, dataloader, config, loss_func, use_tqdm = False, debug=False, plot=False):

    if use_tqdm:
        t = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        t = enumerate(dataloader)
    network.train()  # This is important to call before training!
   
    dev = next(network.parameters()).device  # we are only doing single-device training for now, so this works fine.
    dtype = next(network.parameters()).dtype # we are only doing single-device training for now, so this works fine.
    fix_first_point = config["fix_first_point"]
    bezier_order = network.params_per_dimension-1+int(fix_first_point)
    for (i, imagedict) in t:
        times = imagedict["t"].type(dtype).to(device=dev)
        input_images = imagedict["images"].type(dtype).to(device=dev)

        poses = imagedict["pose"].type(dtype).to(device=dev)
        with torch.no_grad():
            pose_inverses = torch.linalg.inv(poses)
            
            # poses[:,0:3,0:3] = poses[:,0:3,0:3].transpose(1,2)
            # poses[:,0:3,3] = torch.matmul(poses[:,0:3,0:3], -poses[:,0:3,3].unsqueeze(2))[:,:,0]
            # pose_inverses = poses

            raceline_positions_global = (imagedict["positions"]).type(dtype).to(device=dev)
            raceline_positions_global_aug = torch.cat([raceline_positions_global, torch.ones_like(raceline_positions_global[:,:,0]).unsqueeze(2)], dim=2)
            raceline_positions = torch.matmul(raceline_positions_global_aug, pose_inverses[:,0:3].transpose(1,2))

            raceline_velocities_global = (imagedict["velocities"]).type(dtype).to(device=dev)
            raceline_velocities = torch.matmul(raceline_velocities_global, pose_inverses[:,0:3,0:3].transpose(1,2))
            raceline_speeds = torch.norm(raceline_velocities, p=2, dim=2)
            dt = times[:,-1]-times[:,0]
            s = (times - times[:,0,None])/dt[:,None]
            Mpos, controlpoints_fit = deepracing_models.math_utils.bezier.bezierLsqfit(raceline_positions[:,:,[0,1]], bezier_order, t=s)
            lsq_pos = torch.matmul(Mpos, controlpoints_fit)
            Mvel, lsq_v_s = deepracing_models.math_utils.bezier.bezierDerivative(controlpoints_fit, t=s)
            lsq_v_t = lsq_v_s/dt[:,None,None]

        batch_size = input_images.shape[0]
        
        network_output = network(input_images)
        if fix_first_point:
            initial_zeros = torch.zeros(batch_size,1,2,dtype=dtype,device=dev)
            network_output_reshape = network_output.transpose(1,2)
            predictions = torch.cat((initial_zeros,network_output_reshape),dim=1)
        else:
            predictions = network_output.transpose(1,2)        
        pred_points = torch.matmul(Mpos, predictions)

        _, pred_v_s = deepracing_models.math_utils.bezier.bezierDerivative(predictions,  M=Mvel)
        pred_v_t = pred_v_s/dt[:,None,None]

        
        loss = loss_func(pred_points, raceline_positions[:,:,[0,1]])# + 0.1*loss_func(pred_v_t, lsq_v_t)
        
        if debug and config["plot"]:
            a, (b, c) = plt.subplots(1, 2, sharey=False)
            fig : matplotlib.figure.Figure = a
            ax1 : matplotlib.axes.Axes = b
            ax2 : matplotlib.axes.Axes = c
            images_np = np.round(255.0*input_images[0].detach().cpu().numpy().copy().transpose(0,2,3,1)).astype(np.uint8)
            
            ims = []
            for i in range(images_np.shape[0]):
                ims.append([ax1.imshow(images_np[i])])
            ani : animation.ArtistAnimation = animation.ArtistAnimation(fig, ims, interval=250, blit=True, repeat=True)

            xmin = torch.min(raceline_positions[0,:,0]).item() -  10
            xmax = torch.max(raceline_positions[0,:,0]).item() +  10
            ymax = torch.max(torch.abs(raceline_positions[0,:,1])).item() +  2.5

            rlpcpu = raceline_positions.cpu()
            predcpu = pred_points.detach().cpu()
            lsqcpu = lsq_pos.cpu()
            ax2.plot(rlpcpu[0,:,1], rlpcpu[0,:,0], 'g+', label="Ground Truth Waypoints")
            ax2.plot(lsqcpu[0,:,1], lsqcpu[0,:,0], c="b", label="LSQ Fit")
            ax2.plot(predcpu[0,:,1], predcpu[0,:,0], c="r", label="Network Predictions")
            ax2.set_xlim(ymax,-ymax)
            ax2.set_ylim(xmin,xmax)

            plt.show()
            plt.close("all")
        optimizer.zero_grad()
        loss.backward() 
        # Weight and bias updates.
        optimizer.step()
        if use_tqdm:
            t.set_postfix({"current_loss" : loss.item()})
def go(argdict : dict):

    config_file = argdict["model_config_file"]
    dataset_config_file = argdict["dataset_config_file"]
    main_dir = argdict["output_directory"]
    debug = argdict["debug"]
    use_tqdm = argdict["tqdm"]

    with open(dataset_config_file) as f:
        dataset_config : dict = yaml.load(f, Loader = yaml.SafeLoader)
        
    with open(config_file) as f:
        config : dict = yaml.load(f, Loader = yaml.SafeLoader)
    config.update({k : argdict[k] for k in ["debug", "plot", "tqdm"]})
    if argdict["gpu"] is not None:
        config["gpu"] = argdict["gpu"]

    context_length = config["context_length"]
    bezier_order = config["bezier_order"]
    batch_size = config["batch_size"]
    learning_rate = float(config["learning_rate"])
    momentum = float(config["momentum"])
    dampening = float(config["dampening"])
    nesterov = bool(config["nesterov"])
    project_name = config["project_name"]
    fix_first_point = config["fix_first_point"]
    lookahead_time = config["lookahead_time"]
    gpu = config["gpu"] 


    torch.cuda.set_device(gpu)

    num_epochs = config["num_epochs"]
    num_workers = config["num_workers"]
    hidden_dim = config["hidden_dimension"]
    use_3dconv = config["use_3dconv"]
    num_recurrent_layers = config.get("num_recurrent_layers",1)
    config["hostname"] = socket.gethostname()

    
    print("Using config:\n%s" % (str(config)))
    net = deepracing_models.nn_models.Models.AdmiralNetCurvePredictor( input_channels=3, context_length = context_length , hidden_dim = hidden_dim, num_recurrent_layers=num_recurrent_layers, params_per_dimension=bezier_order + 1 - int(fix_first_point), use_3dconv = use_3dconv) 
    print( "net:\n%s" % (str(net),) )
    loss_func : loss_functions.SquaredLpNormLoss = loss_functions.SquaredLpNormLoss()
    use_float = config["use_float"]
    if use_float:
        net = net.float()
    else:
        net = net.double()
    dtype = next(net.parameters()).dtype
    loss_func = loss_func.type(dtype)

    

        
    if gpu>=0:
        print("moving stuff to GPU")
        net = net.cuda(gpu)
        loss_func = loss_func.cuda(gpu)
    optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum = momentum, dampening=dampening, nesterov=nesterov)



    dsets=[]
    dset_tags = dataset_config["tags"]
    dsets_root = dataset_config["root_dir"]
    for dataset in dataset_config["datasets"]:
        dsetdir = os.path.join(dsets_root, dataset["subfolder"])
        sample_count = dataset.get("sample_count", dataset_config.get("sample_count", 160))
        raceline_file = dataset.get("raceline_file", dataset_config.get("raceline_file", None))
        if use_float:
            dsettype = np.float32
        else:
            dsettype = np.float64
        if raceline_file is None:
            current_dset = FD.FutureEgoPoseDataset(dsetdir, dtype=dsettype, context_length=context_length, lookahead_time=lookahead_time, sample_count=sample_count)
            if not ("Ego Behavioral Cloning" in dset_tags):
                dset_tags.append("Ego Behavioral Cloning")
        else:
            current_dset = FD.LocalRacelineDataset(dsetdir, raceline_file, dtype=dsettype, context_length=context_length, lookahead_time=lookahead_time, sample_count=sample_count)
            if not ("Raceline Prediction" in dset_tags):
                dset_tags.append("Raceline Prediction")
        dsets.append(current_dset)

    if len(dsets)==1:
        dset = dsets[0]
    else:
        dset = torch.utils.data.ConcatDataset(dsets)
    
    dataloader = data_utils.DataLoader(dset, batch_size=batch_size, shuffle=True, pin_memory=(gpu>=0), num_workers=num_workers)
    print("Dataloader of of length %d" %(len(dataloader)))
    if debug:
        print("Using datasets:\n%s", (str(dataset_config)))
    
    if debug:
        output_directory = os.path.join(main_dir, "debug")
        os.makedirs(output_directory, exist_ok=True)
        experiment = None
    else:
        experiment = comet_ml.Experiment(workspace="electric-turtle", project_name=project_name)
        output_directory = os.path.join(main_dir, experiment.get_key())
        if os.path.isdir(output_directory):
            raise FileExistsError("%s already exists, this should not happen." %(output_directory) )
        os.makedirs(output_directory)
        experiment.log_parameters(config)
        if len(dset_tags)>0:
            experiment.add_tags(dset_tags)

        config_tags=config.get("tags",[])
        if (type(config_tags)==list) and len(config_tags)>0:
            experiment.add_tags(config_tags)

        experiment_config = {"experiment_key": experiment.get_key()}
        with open(os.path.join(output_directory,"experiment_config.yaml"),"w") as f:
            yaml.dump(experiment_config, stream=f, Dumper=yaml.SafeDumper)
        with open(os.path.join(output_directory,"dataset_config.yaml"), "w") as f:
            yaml.dump(dataset_config, stream=f, Dumper = yaml.SafeDumper)
        with open(os.path.join(output_directory,"model_config.yaml"), "w") as f:
            yaml.dump(config, stream=f, Dumper = yaml.SafeDumper)
        experiment.log_asset(os.path.join(output_directory,"dataset_config.yaml"),file_name="datasets.yaml")
        experiment.log_asset(os.path.join(output_directory,"experiment_config.yaml"),file_name="experiment_config.yaml")
        experiment.log_asset(os.path.join(output_directory,"model_config.yaml"),file_name="model_config.yaml")
        i = 0
    if debug:
        run_epoch(experiment, net, optimizer, dataloader, config, loss_func, debug=True, use_tqdm=use_tqdm)
    else:
        netpostfix = "epoch_%d_params.pt"
        optimizerpostfix = "epoch_%d_optimizer.pt"
        with experiment.train():
            for i in range(num_epochs):
                time.sleep(1.0)
                postfix = i + 1
                modelfile = "params.pt"
                optimizerfile = "optimizer.pt"
                run_epoch(experiment, net, optimizer, dataloader, config, loss_func, use_tqdm=use_tqdm)

                modelout = os.path.join(output_directory,modelfile)
                with open(modelout,'wb') as f:
                    torch.save(net.state_dict(), f)
                optimizerout = os.path.join(output_directory, optimizerfile)
                with open(optimizerout,'wb') as f:
                    torch.save(optimizer.state_dict(), f)
                time.sleep(1.0)

                with open(modelout,'rb') as f:
                    experiment.log_asset( f, file_name=netpostfix %(postfix,) )
                with open(optimizerout,'rb') as f:
                    experiment.log_asset( f, file_name=optimizerpostfix %(postfix,) )

if __name__ == '__main__':
    import logging
    import argparse
    logging.basicConfig()
    parser = argparse.ArgumentParser(description="Train AdmiralNet Pose Predictor")
    parser.add_argument("model_config_file", type=str,  help="Model Configuration file to load")
    parser.add_argument("dataset_config_file", type=str,  help="Dataset Configuration file to load")
    parser.add_argument("output_directory", type=str, help="Where to save models.")
    parser.add_argument("--debug", action="store_true",  help="Don't actually push to comet, just testing")
    parser.add_argument("--plot", action="store_true",  help="Don't actually push to comet, just testing")
    parser.add_argument("--tqdm", action="store_true",  help="Display tqdm progress bar on each epoch")
    parser.add_argument("--gpu", type=int, default=None,  help="Override the GPU index specified in the config file")
    args = parser.parse_args()
    argdict : dict = vars(args)
    go(argdict)    
    