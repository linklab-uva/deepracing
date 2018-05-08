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
def main():
    parser = argparse.ArgumentParser(description="Deepf1 playground")
    parser.add_argument("--x", type=int, default=25,  help="X coordinate for where to put the center of the steering wheel")
    parser.add_argument("--y", type=int, default=25,  help="Y coordinate for where to put the center of the steering wheel")
    parser.add_argument("--wheelrows", type=int, default=50,  help="Number of rows to resize the steering wheel to")
    parser.add_argument("--wheelcols", type=int, default=50,  help="Number of columns to resize the steering wheel to")
    parser.add_argument("--max_angle", type=float, default=180.0,\
          help="Maximum angle that the scaled annotations represent")
    parser.add_argument("--output_folder", type=str, default='prediction_images',\
          help="Output video file")
    parser.add_argument("--output_video", type=str, default='annotated_video.avi',\
          help="Output video file")
    parser.add_argument("--model_file", type=str, required=True,  help="Model weights to load from file")
    parser.add_argument("--root_dir", type=str, required=True, help="Root dir of the F1 validation set to use")
    parser.add_argument("--annotation_file", type=str, required=True, help="Annotation file to use")
    parser.add_argument("--plot", action="store_true", help="Plot some statistics of the results")
    parser.add_argument("--label_scale", type=float, default=100.0, help="Scaling factor that was used during training so that scaling can be un-done at test time")
    parser.add_argument("--im_scale", type=float, default=1.0, help="Image Scaling factor that was used during training so that scaling can be un-done at test time")
    args = parser.parse_args()
    prefix, ext = args.annotation_file.split(".")

    output_video = args.output_video
    wheelrows = args.wheelrows
    wheelcols = args.wheelcols
    x = int(args.x-wheelrows/2)
    y = int(args.y-wheelcols/2)
    max_angle = args.max_angle
    wheel = cv2.imread('steering_wheel.png', cv2.IMREAD_UNCHANGED)  
    wheel =  cv2.resize(wheel, (wheelrows, wheelcols)) 
    input_folder = os.path.join(args.root_dir,'raw_images')
    if(not os.path.isdir(args.output_folder)):
        os.makedirs(args.output_folder)
    annotations_path = os.path.join(args.root_dir,args.annotation_file)
    annotations_file = open(annotations_path,'r')
    annotations = annotations_file.readlines()
    predictions = []
    ground_truths = []
    filename, _, anglestr, _, _ = annotations[0].split(",")
    img_path = os.path.join(input_folder,filename)
    background = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    size = background.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_vid = os.path.join(args.output_folder,output_video)
    videoout = cv2.VideoWriter(output_vid ,fourcc, 60.0, (size[1], size[0]),True)
    network = models.PilotNet()
    network.float()
    network.cuda()
    state_dict = torch.load(args.model_file)
    network.load_state_dict(state_dict)
    network.eval()
    prefix, ext = args.annotation_file.split(".")
    trainset = loaders.F1Dataset(args.root_dir, args.annotation_file, (66,200), use_float32=True)
    '''
    trainset.read_files()
    '''
    if((not os.path.isfile("./" + prefix+"_images.pkl")) or (not os.path.isfile("./" + prefix+"_annotations.pkl"))):
        trainset.read_files()
        trainset.write_pickles(prefix+"_images.pkl",prefix+"_annotations.pkl")
    else:  
        trainset.read_pickles(prefix+"_images.pkl",prefix+"_annotations.pkl")
    mean,stdev = trainset.statistics()
    print(mean)
    print(stdev)
    img_transformation = transforms.Compose([transforms.Normalize(mean,stdev)])
    trainset.img_transformation = img_transformation
    loader = torch.utils.data.DataLoader(trainset, batch_size = 1, shuffle = False, num_workers = 0)
    cum_diff = 0.0
    t = tqdm(enumerate(loader))
    for idx,(inputs, labels) in t:
        inputs = inputs.cuda()
        pred = network(inputs)
        angle = pred.item()/args.label_scale
        ground_truth = float(anglestr.replace("\n",""))
        scaled_ground_truth = max_angle * labels.item()
        scaled_angle = max_angle * angle
        predictions.append(scaled_angle)
        ground_truths.append(scaled_ground_truth)
        t.set_postfix(scaled_angle = scaled_angle, scaled_ground_truth = scaled_ground_truth)
       # print("Ground Truth: %f. Prediction: %f.\n" %(scaled_ground_truth, scaled_angle))
        '''
        M = cv2.getRotationMatrix2D((wheelrows/2,wheelcols/2),scaled_angle,1)
        wheel_rotated = cv2.warpAffine(wheel,M,(wheelrows,wheelcols))
        overlayed = imutils.annotation_utils.overlay_image(background,wheel_rotated,x,y)
        name, _ = filename.split(".")
        _,_,img_num_str = name.split("_")
        img_num_str = img_num_str.replace("\n","")
        output_path = os.path.join(args.output_folder,'overlayed_image_' + img_num_str + ".jpg")
        cv2.imwrite(output_path,overlayed)
        videoout.write(overlayed)
        '''
    predictions_array = np.array(predictions)
    ground_truths_array = np.array(ground_truths)
    diffs = np.subtract(predictions_array,ground_truths_array)
    rms = np.mean(np.square(diffs))
    print("RMS Error: ", rms)
    if args.plot:
        from scipy import stats
        import matplotlib.pyplot as plt
        t = np.linspace(0,len(loader)-1,len(loader))
        plt.plot(t,predictions_array,'r')
        plt.plot(t,ground_truths_array,'b')
        plt.plot(t,diffs,'g')
        plt.show()
if __name__ == '__main__':
    main()
