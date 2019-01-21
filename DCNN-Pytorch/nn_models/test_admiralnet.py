import cv2
import glob
import numpy as np
import nn_models
import data_loading.image_loading as il
import nn_models.Models as models
import data_loading.data_loaders as loaders
import numpy.random
import torch, random
import torch.nn as nn 
import torch.optim as optim
from tqdm import tqdm as tqdm
import pickle
from datetime import datetime
import os
import string
import argparse
from random import randint
from datetime import datetime
import imutils.annotation_utils
from data_loading.image_loading import load_image
import torchvision.transforms as transforms
from scipy import stats
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Test AdmiralNet")
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--annotation_file", type=str, required=True)
    parser.add_argument("--write_images", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    
    annotation_dir, annotation_file = os.path.split(args.annotation_file)
    model_dir, model_file = os.path.split(args.model_file)
    config_path = os.path.join(model_dir,'config.pkl')
    config_file = open(config_path,'rb')
    config = pickle.load(config_file)
    print(config)
    model_prefix, _ = model_file.split(".")
   # return

    gpu = int(config['gpu'])
    use_float32 = bool(config['use_float32'])
    label_scale = float(config['label_scale'])
    size = (66,200)
    prefix, _ = annotation_file.split(".")
    prefix = prefix + config['file_prefix']
    context_length = int(config['context_length'])
    sequence_length = int(config['sequence_length'])
    hidden_dim = int(config.get('hidden_dim','100'))
    #optical_flow = bool(config.get('optical_flow',''))
    rnn_cell_type='lstm'
    network = models.AdmiralNet(cell=rnn_cell_type, context_length = context_length, sequence_length=sequence_length, hidden_dim = hidden_dim, gpu = gpu)
    state_dict = torch.load(args.model_file)
    network.load_state_dict(state_dict)
    network.projector_input = torch.load(  open(os.path.join(model_dir,"projector_input.pt"), 'r+b') )
    network.init_hidden = torch.load(  open(os.path.join(model_dir,"init_hidden.pt"), 'r+b') )
    network.init_cell = torch.load(  open(os.path.join(model_dir,"init_cell.pt"), 'r+b') )
    network = network.float()
    print(network)
    valset = loaders.F1OpticalFlowDataset(args.annotation_file, size, context_length = context_length, sequence_length = sequence_length)    
    if(gpu>=0):
        network = network.cuda(gpu)
    
    annotation_prefix = annotation_file.split(".")[0]
    image_pickle = os.path.join( annotation_dir, annotation_prefix + "_flow_images.pt")
    labels_pickle = os.path.join( annotation_dir, annotation_prefix + "_flow_labels.pt")
    if(os.path.isfile(image_pickle) and os.path.isfile(labels_pickle)):
        valset.loadPickles()
    else:  
        valset.loadFiles()
        valset.writePickles()

    predictions=[]
    ground_truths=[]
    losses=[]
    criterion = nn.MSELoss()
    cum_diff = 0.0
    if(gpu>=0):
        criterion = criterion.cuda(gpu)
    network.eval()
    loader = torch.utils.data.DataLoader(valset, batch_size = 1, shuffle = False, num_workers = 0)
    
    t = tqdm(enumerate(loader))
    for idx,(inputs,labels) in t:
        if(gpu>=0):
            inputs = inputs.cuda(gpu)
            labels = labels.cuda(gpu)
        pred = network(inputs)
        #result_data.append([labels,pred])
        print(pred.shape)
        angle = pred[0][0][0]
        ground_truth = labels[0][0][0]
        predictions.append(angle)
        ground_truths.append(ground_truth)
        loss = criterion(pred, labels)
        losses.append(loss.item())
        t.set_postfix(angle = angle, ground_truth = ground_truth)

    predictions_array = np.array(predictions)
    ground_truths_array = np.array(ground_truths)
    # log_name = "ouput_log.txt"
    # imdir = "admiralnet_prediction_images_" + model_prefix
    # if(os.path.exists(imdir)==False):
    #     os.mkdir(imdir)
    # log_output_path = os.path.join(imdir,log_name)
    # log = list(zip(ground_truths_array,predictions_array))
    # with open(log_output_path, "a") as myfile:
    #     for x in log:
    #         log_item = [x[0],x[1]]
    #         myfile.write("{0},{1}\n".format(log_item[0],log_item[1]))
    diffs = np.subtract(predictions_array,ground_truths_array)
    rms = np.sqrt(np.mean(np.array(losses)))
    nrms = np.sqrt(np.mean(np.divide(np.square(np.array(losses)),np.multiply(np.mean(np.array(predictions)),np.mean(np.array(ground_truths))))))
    print("RMS Error: ", rms)
    print("NRMS Error: ", nrms)

    if args.plot:
        fig = plt.figure()
        ax = plt.subplot(111)
        t = np.linspace(0,len(predictions_array)-1,len(predictions_array))
        ax.plot(t,predictions_array,'r',label='Predicted')
        ax.plot(t,ground_truths_array,'b',label='Ground Truth')
        ax.legend()
        plt.savefig("admiralnet_prediction_images_" + model_prefix+"\plot.jpeg")
        plt.show()
if __name__ == '__main__':
    main()
