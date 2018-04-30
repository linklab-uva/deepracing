from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python.net_builder import ops
from caffe2.python import core, workspace, model_helper, utils, brew
import urllib
from caffe2.python.rnn_cell import LSTM
from caffe2.proto import caffe2_pb2
from tqdm import tqdm as tqdm
import img_utils.utils as imutils
import caffe2.python.predictor.predictor_py_utils as pred_utils
from PilotNet import PilotNet
import cv2
import argparse
import logging
import os
from random import randint
import numpy as np
from datetime import datetime
logging.basicConfig()
log = logging.getLogger("PilotNet")
log.setLevel(logging.ERROR)
def main():
    parser = argparse.ArgumentParser(description="Deepf1 playground")
    parser.add_argument("--x", type=int, default=25,  help="X coordinate for where to put the center of the steering wheel")
    parser.add_argument("--y", type=int, default=25,  help="Y coordinate for where to put the center of the steering wheel")
    parser.add_argument("--wheelrows", type=int, default=50,  help="Number of rows to resize the steering wheel to")
    parser.add_argument("--wheelcols", type=int, default=50,  help="Number of columns to resize the steering wheel to")
    parser.add_argument("--max_angle", type=float, default=180.0,\
          help="Maximum angle that the scaled annotations represent")
    parser.add_argument("--output_video", type=str, default='annotated_video.avi',\
          help="Output video file")
    args = parser.parse_args()
    output_video = args.output_video
    wheelrows = args.wheelrows
    wheelcols = args.wheelcols
    x = int(args.x-wheelrows/2)
    y = int(args.y-wheelcols/2)
    max_angle = args.max_angle
    SCALE_FACTOR=2.55
    INIT_NET = "init_net_linear_interpolation.pb"
    PREDICT_NET = "predict_net_linear_interpolation.pb"
    device = device_option=core.DeviceOption(caffe2_pb2.CUDA,0)
    init_def = caffe2_pb2.NetDef()
    with open(INIT_NET, 'rb') as f:
        init_def.ParseFromString(f.read())
        init_def.device_option.CopyFrom(device)
    workspace.RunNetOnce(init_def.SerializeToString())
    #print(init_def)
    net_def = caffe2_pb2.NetDef()
    with open(PREDICT_NET, 'rb') as f:
        net_def.ParseFromString(f.read())
        net_def.device_option.CopyFrom(device)
        workspace.CreateNet(net_def.SerializeToString())
    #`#run the net and return prediction
    #print(net_def)

    annotations_path = os.path.join('D:/test_data/slow_run_australia_track2','linear_interpolation.csv')
    annotations_file = open(annotations_path,'r')
    wheel = cv2.imread('steering_wheel.png', cv2.IMREAD_UNCHANGED)  
    wheel =  cv2.resize(wheel, (wheelrows, wheelcols)) 
    input_folder = os.path.join('D:/test_data/slow_run_australia_track2','raw_images')
    output_folder = os.path.join('D:/test_data/slow_run_australia_track2','prediction_images')
    annotations = annotations_file.readlines()
    filename, _, anglestr = annotations[0].split(",")
    img_path = os.path.join(input_folder,filename)
    background = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    size = background.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_vid = os.path.join('D:/test_data/slow_run_australia_track2','prediction_images',output_video)
    videoout = cv2.VideoWriter(output_vid ,fourcc, 60.0, (size[1], size[0]),True)
    for annotation in tqdm(annotations):
        filename, ts, anglestr = annotation.split(",")
        img_path = os.path.join(input_folder,filename)
        background = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_resized= cv2.resize(background,dsize=(200,66), interpolation = cv2.INTER_CUBIC)
        img_transposed = np.transpose(img_resized, (2, 0, 1)).astype(np.float32)
        input = np.random.rand(1,3,66,200).astype(np.float32)
        input[0] = img_transposed
        input = np.divide(input, SCALE_FACTOR)
        workspace.FeedBlob('input_blob', input, device_option=device)
        workspace.RunNet("PilotNet_1")
        pred = workspace.FetchBlob("prediction")
        pred_scaled = np.divide(pred,100.0)
        angle = float(pred_scaled)
        scaled_angle = max_angle * angle
      #  print(scaled_angle)
        M = cv2.getRotationMatrix2D((wheelrows/2,wheelcols/2),scaled_angle,1)
        wheel_rotated = cv2.warpAffine(wheel,M,(wheelrows,wheelcols))
        overlayed = imutils.overlay_image(background,wheel_rotated,x,y)
        name, _ = filename.split(".")
        _,_,img_num_str = name.split("_")
        img_num_str = img_num_str.replace("\n","")
        output_path = os.path.join(output_folder,'overlayed_image_' + img_num_str + ".jpg")
        cv2.imwrite(output_path,overlayed)
        videoout.write(overlayed)
if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()
