import comet_ml
import torch
import torch.nn as NN
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
import deepracing.backend
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import deepracing_models.math_utils.bezier
import socket

#torch.backends.cudnn.enabled = False
def run_epoch(experiment, network, optimizer, trainLoader, gpu, params_loss, kinematic_loss, loss_weights, imsize=(66,200), timewise_weights=None, debug=False, use_tqdm=True):
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
    # loss_weights_torch = torch.tensor(loss_weights)
    # if use_float:
    #     loss_weights_torch = session_times_torch.loss_weights_torch()
    # else:
    #     loss_weights_torch = loss_weights_torch.double()
    # if gpu>=0:
    #     loss_weights_torch = loss_weights_torch.cuda(gpu)
    for (i, (image_torch, opt_flow_torch, positions_torch, quats_torch, linear_velocities_torch, angular_velocities_torch, session_times_torch) ) in t:
        if network.input_channels==5:
            image_torch = torch.cat((image_torch,opt_flow_torch),axis=2)
        if gpu>=0:
            image_torch = image_torch.cuda(gpu)
            positions_torch = positions_torch.cuda(gpu)
            session_times_torch = session_times_torch.cuda(gpu)
            linear_velocities_torch = linear_velocities_torch.cuda(gpu)
        predictions = network(image_torch)
        dt = session_times_torch[:,-1]-session_times_torch[:,0]
        s_torch = (session_times_torch - session_times_torch[:,0,None])/dt[:,None]
        fitpoints = positions_torch[:,:,[0,2]]
        fitvels = linear_velocities_torch[:,:,[0,2]]
        bezier_order = network.params_per_dimension-1
        Mfit, controlpoints_fit = deepracing_models.math_utils.bezier.bezierLsqfit(fitpoints,s_torch,bezier_order)
        predictions_reshape = predictions.transpose(1,2)
        pred_points = torch.matmul(Mfit, predictions_reshape)
        _, pred_vels = deepracing_models.math_utils.bezier.bezierDerivative(predictions_reshape,s_torch)
        if debug:
            images_np = image_torch[0].detach().cpu().numpy().copy()
            num_images = images_np.shape[0]
            print(num_images)
            images_np_transpose = np.zeros((num_images, images_np.shape[2], images_np.shape[3], images_np.shape[1]), dtype=np.uint8)
            ims = []
            for i in range(num_images):
                images_np_transpose[i]=skimage.util.img_as_ubyte(images_np[i].transpose(1,2,0))
                im = plt.imshow(images_np_transpose[i], animated=True)
                ims.append([im])
            ani = animation.ArtistAnimation(plt.figure(), ims, interval=250, blit=True, repeat_delay=2000)
            fig = plt.figure()
            ax = fig.add_subplot()
            fitpointsnp = fitpoints[0,:].detach().cpu().numpy().copy()
            ax.plot(fitpointsnp[:,0],fitpointsnp[:,1],'r-')
            
            evalpoints = torch.matmul(Mfit, controlpoints_fit)
            evalpointsnp = evalpoints[0,:].detach().cpu().numpy().copy()
            ax.plot(evalpointsnp[:,0],evalpointsnp[:,1],'bo')
            plt.show()

        current_param_loss = params_loss(predictions_reshape,controlpoints_fit)
        current_position_loss = kinematic_loss(pred_points,fitpoints)
        current_velocity_loss = kinematic_loss(pred_vels/dt[:,None,None],fitvels)
        loss = loss_weights[0]*current_param_loss + loss_weights[1]*current_position_loss + loss_weights[2]*current_velocity_loss
        
        # Backward pass:
        optimizer.zero_grad()
        loss.backward() 
        # Weight and bias updates.
        optimizer.step()
        # logging information
        cum_loss += float(loss.item())
        cum_param_loss += float(current_param_loss.item())
        cum_position_loss += float(current_position_loss.item())
        cum_velocity_loss += float(current_velocity_loss.item())
        num_samples += 1.0
        experiment.log_metric("cumulative_position_error", cum_position_loss/num_samples, step=i)
        experiment.log_metric("cumulative_velocity_error", cum_velocity_loss/num_samples, step=i)
        if use_tqdm:
            t.set_postfix({"cum_loss" : cum_loss/num_samples,"cum_param_loss" : cum_param_loss/num_samples,"cum_position_loss" : cum_position_loss/num_samples,"cum_velocity_loss" : cum_velocity_loss/num_samples})
def go():
    parser = argparse.ArgumentParser(description="Train AdmiralNet Pose Predictor")
    parser.add_argument("dataset_config_file", type=str,  help="Dataset Configuration file to load")
    parser.add_argument("model_config_file", type=str,  help="Model Configuration file to load")
    parser.add_argument("output_directory", type=str,  help="Where to put the resulting model files")

    parser.add_argument("--context_length",  type=int, default=None,  help="Override the context length specified in the config file")
    parser.add_argument("--epochstart", type=int, default=1,  help="Restart training from the given epoch number")
    parser.add_argument("--debug", action="store_true",  help="Display images upon each iteration of the training loop")
    parser.add_argument("--override", action="store_true",  help="Delete output directory and replace with new data")
    parser.add_argument("--tqdm", action="store_true",  help="Display tqdm progress bar on each epoch")
    parser.add_argument("--batch_size", type=int, default=None,  help="Override the order of the batch size specified in the config file")
    parser.add_argument("--gpu", type=int, default=None,  help="Override the GPU index specified in the config file")
    parser.add_argument("--learning_rate", type=float, default=None,  help="Override the learning rate specified in the config file")
    parser.add_argument("--bezier_order", type=int, default=None,  help="Override the order of the bezier curve specified in the config file")
    parser.add_argument("--weighted_loss", action="store_true",  help="Use timewise weights on param loss")
    parser.add_argument("--adam", action="store_true",  help="Use ADAM instead of SGD")
    parser.add_argument("--rmsprop", action="store_true",  help="Use RMSprop instead of SGD")
    

    args = parser.parse_args()
    dataset_config_file = args.dataset_config_file
    config_file = args.model_config_file
    debug = args.debug
    epochstart = args.epochstart
    weighted_loss = args.weighted_loss
    with open(config_file) as f:
        config = yaml.load(f, Loader = yaml.SafeLoader)

    with open(dataset_config_file) as f:
        dataset_config = yaml.load(f, Loader = yaml.SafeLoader)

    image_size = dataset_config["image_size"]
    input_channels = config["input_channels"]
    
    if args.context_length is not None:
        context_length = args.context_length
        config["context_length"]  = context_length
        output_directory+="_context%d"%(context_length)
    else:
        context_length = config["context_length"]
        
    if args.bezier_order is not None:
        bezier_order = args.bezier_order
        config["bezier_order"]  = bezier_order
        output_directory+="_bezier_order%d"%(bezier_order)
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
    momentum = config["momentum"]
    num_epochs = config["num_epochs"]
    num_workers = config["num_workers"]
    use_float = config["use_float"]
    loss_weights = config["loss_weights"]
    hidden_dim = config["hidden_dimension"]
    num_recurrent_layers = config.get("num_recurrent_layers",1)
    config["hostname"] = socket.gethostname()
    
    
    print("Using config:\n%s" % (str(config)))
    net = deepracing_models.nn_models.Models.AdmiralNetCurvePredictor( context_length = context_length , input_channels=input_channels, hidden_dim = hidden_dim, num_recurrent_layers=num_recurrent_layers, params_per_dimension=bezier_order+1 ) 
    print("net:\n%s" % (str(net)))
    ppd = net.params_per_dimension
    numones = int(ppd/2)
    if weighted_loss:
        timewise_weights = torch.from_numpy( np.hstack( ( np.ones(numones), np.linspace(1,3, ppd - numones ) ) ) )
    else:
        timewise_weights = None
    params_loss = deepracing_models.nn_models.LossFunctions.SquaredLpNormLoss(timewise_weights=timewise_weights)
    kinematic_loss = deepracing_models.nn_models.LossFunctions.SquaredLpNormLoss()
    if use_float:
        print("casting stuff to float")
        net = net.float()
        params_loss = params_loss.float()
        kinematic_loss = kinematic_loss.float()
    else:
        print("casting stuff to double")
        net = net.double()
        params_loss = params_loss.double()
        kinematic_loss = kinematic_loss.float()
    if gpu>=0:
        print("moving stuff to GPU")
        net = net.cuda(gpu)
        params_loss = params_loss.cuda(gpu)
        kinematic_loss = kinematic_loss.cuda(gpu)
    if args.adam:
        config["optimizer"] = "Adam"
        optimizer = optim.Adam(net.parameters(), lr = learning_rate)
    elif args.rmsprop:
        config["optimizer"] = "RMSprop"
        optimizer = optim.RMSprop(net.parameters(), lr = learning_rate, momentum = momentum)
    else:
        config["optimizer"] = "SGD"
        optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum = momentum, dampening=0.000, nesterov=True)
    netpostfix = "epoch_%d_params.pt"
    optimizerpostfix = "epoch_%d_optimizer.pt"
    
    main_dir = args.output_directory
    experiment = comet_ml.Experiment(workspace="electric-turtle", project_name="deepracingbezierpredictor")
    experiment.log_parameters(config)
    experiment.log_parameters(dataset_config)
    experiment.add_tag("bezierpredictor")
    experiment_config = {"experiment_key": experiment.get_key()}
    output_directory = os.path.join(main_dir, experiment.get_key())
    if os.path.isdir(output_directory) :
        raise FileExistsError("%s already exists, this should not happen." %(output_directory) )
    os.makedirs(output_directory)
    yaml.dump(experiment_config, stream=open(os.path.join(output_directory,"experiment_config.yaml"),"w"), Dumper=yaml.SafeDumper)
    yaml.dump(dataset_config, stream=open(os.path.join(output_directory,"dataset_config.yaml"), "w"), Dumper = yaml.SafeDumper)
    yaml.dump(config, stream=open(os.path.join(output_directory,"model_config.yaml"), "w"), Dumper = yaml.SafeDumper)
    
    
        
    if num_workers == 0:
        max_spare_txns = 50
    else:
        max_spare_txns = num_workers

    #image_wrapper = deepracing.backend.ImageFolderWrapper(os.path.dirname(image_db))
    
    dsets=[]
    use_optflow = net.input_channels==5
    for dataset in dataset_config["datasets"]:
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


        curent_dset = PD.PoseSequenceDataset(image_wrapper, label_wrapper, key_file, context_length,\
                     image_size = image_size, return_optflow=use_optflow, apply_color_jitter=apply_color_jitter, erasing_probability=erasing_probability)
        dsets.append(curent_dset)
        print("\n")
    if len(dsets)==1:
        dset = dsets[0]
    else:
        dset = torch.utils.data.ConcatDataset(dsets)
    
    dataloader = data_utils.DataLoader(dset, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers, pin_memory=gpu>=0)
    print("Dataloader of of length %d" %(len(dataloader)))
    if(epochstart==1):
        i = 0
    else:
        i = epochstart
    with experiment.train():
        while i < num_epochs:
            time.sleep(2.0)
            postfix = i + 1
            print("Running Epoch Number %d" %(postfix))
            #dset.clearReaders()
            try:
                tick = time.time()
                run_epoch(experiment, net, optimizer, dataloader, gpu, params_loss, kinematic_loss, loss_weights, debug=debug, use_tqdm=args.tqdm )
                tock = time.time()
                print("Finished epoch %d in %f seconds." % ( postfix , tock-tick ) )
                experiment.log_epoch_end(postfix)
            except Exception as e:
                if isinstance(e, FileExistsError):
                    print(e)
                    exit(-1)
                print("Restarting epoch %d because %s"%(postfix, str(e)))
                modelin = os.path.join(output_directory, netpostfix %(postfix-1))
                optimizerin = os.path.join(output_directory,optimizerpostfix %(postfix-1))
                net.load_state_dict(torch.load(modelin))
                optimizer.load_state_dict(torch.load(optimizerin))
                continue
            modelout = os.path.join(output_directory,netpostfix %(postfix))
            torch.save(net.state_dict(), modelout)
            with open(modelout,'rb') as modelfile:
                experiment.log_asset(modelfile,file_name=netpostfix %(postfix))
            optimizerout = os.path.join(output_directory,optimizerpostfix %(postfix))
            torch.save(optimizer.state_dict(), optimizerout)
            with open(optimizerout,'rb') as optimizerfile:
                experiment.log_asset(optimizerfile,file_name=optimizerpostfix %(postfix))
            i = i + 1
import logging
if __name__ == '__main__':
    logging.basicConfig()
    go()    
    