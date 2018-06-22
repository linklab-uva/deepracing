import cv2
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
    hidden_dim = int(config['hidden_dim'])
    optical_flow = bool(config.get('optical_flow',''))
    rnn_cell_type='lstm'
    network = models.AdmiralNet(cell=rnn_cell_type,context_length = context_length, sequence_length=sequence_length, hidden_dim = hidden_dim, use_float32 = use_float32, gpu = gpu, optical_flow=optical_flow)
    state_dict = torch.load(args.model_file)
    network.load_state_dict(state_dict)
    print(network)
    #result_data=[]
    if(label_scale == 1.0):
        label_transformation = None
    else:
        label_transformation = transforms.Compose([transforms.Lambda(lambda inputs: inputs.mul(label_scale))])
    if(use_float32):
        network.float()
        trainset = loaders.F1SequenceDataset(annotation_dir,annotation_file,(66,200),\
        context_length=context_length, sequence_length=sequence_length, use_float32=True, label_transformation = label_transformation, optical_flow=optical_flow)
    else:
        network.double()
        trainset = loaders.F1SequenceDataset(annotation_dir, annotation_file,(66,200),\
        context_length=context_length, sequence_length=sequence_length, label_transformation = label_transformation, optical_flow=optical_flow)
    
    if(gpu>=0):
        network = network.cuda(gpu)
    if optical_flow:
        if((not os.path.isfile("./" + prefix+"_opticalflows.pkl")) or (not os.path.isfile("./" + prefix+"_opticalflowannotations.pkl"))):
            trainset.read_files_flow()
            trainset.write_pickles(prefix+"_opticalflows.pkl",prefix+"_opticalflowannotations.pkl")
        else:  
            trainset.read_pickles(prefix+"_opticalflows.pkl",prefix+"_opticalflowannotations.pkl")
    else:
        if((not os.path.isfile("./" + prefix+"_images.pkl")) or (not os.path.isfile("./" + prefix+"_annotations.pkl"))):
            trainset.read_files()
            trainset.write_pickles(prefix+"_images.pkl",prefix+"_annotations.pkl")
        else:  
            trainset.read_pickles(prefix+"_images.pkl",prefix+"_annotations.pkl")

    trainset.img_transformation = config['image_transformation']
    loader = torch.utils.data.DataLoader(trainset, batch_size = 1, shuffle = False, num_workers = 0)
    cum_diff = 0.0
    t = tqdm(enumerate(loader))
    network.eval()
    predictions=[]
    ground_truths=[]
    losses=[]
    criterion = nn.MSELoss()
    if(gpu>=0):
        criterion = criterion.cuda(gpu)
    if args.write_images:
        imdir = "admiralnet_prediction_images_" + model_prefix
        os.mkdir(imdir)
        annotation_file = open(args.annotation_file,'r')
        annotations = annotation_file.readlines()
        annotation_file.close()
        im,_,_,_,_ = annotations[0].split(",")
        background = cv2.imread(os.path.join(annotation_dir,'raw_images_full',im),cv2.IMREAD_UNCHANGED)
        out_size = background.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        videoout = cv2.VideoWriter(os.path.join(imdir,"video.avi") ,fourcc, 30.0, (out_size[1], out_size[0]),True)
        wheel_pred = cv2.imread('predicted_fixed.png',cv2.IMREAD_UNCHANGED)
        wheelrows_pred = 65
        wheelcols_pred = 65
        wheel_pred = cv2.resize(wheel_pred, (wheelcols_pred,wheelrows_pred), interpolation = cv2.INTER_CUBIC)
    for idx,(inputs, throttle, brake,_, labels) in t:
        if(gpu>=0):
            inputs = inputs.cuda(gpu)
            throttle = throttle.cuda(gpu)
            brake= brake.cuda(gpu)
            labels = labels.cuda(gpu)
        pred = torch.div(network(inputs,throttle,brake),label_scale)
        #result_data.append([labels,pred])
        if pred.shape[1] == 1:
            angle = pred.item()
            ground_truth = labels.item()
        else:
            angle = pred.squeeze()[0].item()
            ground_truth = labels.squeeze()[0].item()
        predictions.append(angle)
        ground_truths.append(ground_truth)
        loss = criterion(pred, labels)
        losses.append(loss.item())
        t.set_postfix(angle = angle, ground_truth = ground_truth)
        #print("Ground Truth: %f. Prediction: %f.\n" %(scaled_ground_truth, scaled_angle))
        if args.write_images:
            scaled_pred_angle = 180.0*angle
            M_pred = cv2.getRotationMatrix2D((wheelrows_pred/2,wheelcols_pred/2),scaled_pred_angle,1)
            wheel_pred_rotated = cv2.warpAffine(wheel_pred,M_pred,(wheelrows_pred,wheelcols_pred))
            numpy_im = np.transpose(trainset.images[idx],(1,2,0)).astype(np.float32)
           # print(numpy_im.shape)
            im,_,_,_,_ = annotations[idx].split(",")
            background = cv2.imread(os.path.join(annotation_dir,'raw_images_full',im),cv2.IMREAD_UNCHANGED)
            out_size = background.shape

            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (int((out_size[1]-wheelcols_pred)/2)-15,int((out_size[0]-wheelcols_pred)/2)-35)
            fontScale              = 0.45
            fontColor              = (0,0,0)
            lineType               = 1
            
            overlay = background.copy()
            cv2.rectangle(overlay, (int((out_size[1]-wheelcols_pred)/2)-20,int((out_size[0]-wheelcols_pred)/2)-33), (int((out_size[1]-wheelcols_pred)/2)+90,int((out_size[0]-wheelcols_pred)/2)-47),(255, 255, 255,0.2), -1)

            alpha=0.5
            cv2.addWeighted(overlay, alpha, background, 1 - alpha,0, background)

            cv2.putText(background,'Predicted:' + "{0:.2f}".format(angle),bottomLeftCornerOfText,font,fontScale,fontColor,lineType)

            overlayed_pred = imutils.annotation_utils.overlay_image(background,wheel_pred_rotated,int((out_size[1]-wheelcols_pred)/2),int((out_size[0]-wheelcols_pred)/2)-120)
            
            name = "ouput_image_" + str(idx) + ".png"
            output_path = os.path.join(imdir,name)
            cv2.imwrite(output_path,overlayed_pred)
            videoout.write(overlayed_pred)
    predictions_array = np.array(predictions)
    ground_truths_array = np.array(ground_truths)
    log_name = "ouput_log.txt"
    imdir = "admiralnet_prediction_images_" + model_prefix
    if(os.path.exists(imdir)==False):
        os.mkdir(imdir)
    log_output_path = os.path.join(imdir,log_name)
    log = list(zip(ground_truths_array,predictions_array))
    with open(log_output_path, "a") as myfile:
        for x in log:
            log_item = [x[0],x[1]]
            myfile.write("{0},{1}\n".format(log_item[0],log_item[1]))
    diffs = np.subtract(predictions_array,ground_truths_array)
    rms = np.sqrt(np.mean(np.array(losses)))
    nrms = np.sqrt(np.mean(np.divide(np.square(np.array(losses)),np.dot(np.mean(np.array(predictions)),np.mean(np.array(ground_truths))))))
    print("RMS Error: ", rms)
    print("NRMS Error: ", nrms)

    if args.plot:
        fig = plt.figure()
        ax = plt.subplot(111)
        t = np.linspace(0,len(loader)-1,len(loader))
        ax.plot(t,predictions_array,'r',label='Predicted')
        ax.legend()
        plt.savefig("admiralnet_prediction_images_" + model_prefix+"\plot.jpeg")
        plt.show()
if __name__ == '__main__':
    main()
