from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python.net_builder import ops
from caffe2.python import core, workspace, model_helper, utils, brew
from caffe2.python.rnn_cell import LSTM
from caffe2.proto import caffe2_pb2
from caffe2.python.optimizer import build_sgd
from caffe2.python.predictor.mobile_exporter import Export, add_tensor
from caffe2.python.predictor.predictor_exporter import get_predictor_exporter_helper, PredictorExportMeta


import caffe2.python.predictor.predictor_py_utils as pred_utils

from caffe2.python.predictor_constants import predictor_constants as predictor_constants
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
# Default set() here is intentional as it would accumulate values like a global
# variable
def CreateNetOnce(net, created_names=set()): # noqa
    name = net.Name()
    if name not in created_names:
        created_names.add(name)
        workspace.CreateNet(net)
class PilotNet(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.output_dim = args.output_dim
        self.scale_factor = args.scale_factor
        self.input_channels = 3
        self.input_height = 66
        self.input_width = 200
        self.num_training_iterations = args.num_iterations
        self.input_file = args.input_file
        self.input_folder = args.input_folder
    def CreateModel(self):
        log.debug("Start training")
        model = model_helper.ModelHelper(name="PilotNet")
        input_blob =  model.net.AddExternalInputs('input_blob')
        #3x3 ->Convolutional feature map 64@1x18
        #3x3 ->Convolutional feature map 64@3x20
        #5x5 ->Convolutional feature map 48@5x22
        #5x5 ->Convolutional feature map 36@14x47
        #5x5 -> Convolutional feature map 24@31x98
        #Normalized input planes 3@66x200
	    # Image size: 66x200 -> 31x98
        self.conv1 = brew.conv(model, input_blob, 'conv1', dim_in=self.input_channels, dim_out=24, kernel=5, stride=2)
	    # Image size: 31x98 -> 14x47
        self.conv2 = brew.conv(model, self.conv1, 'conv2', dim_in=24, dim_out=36, kernel=5, stride=2)
	    # Image size: 14x47 -> 5x22
        self.conv3 = brew.conv(model, self.conv2, 'conv3', dim_in=36, dim_out=48, kernel=5, stride=2)
	    # Image size: 5x22 -> 3x20
        self.conv4 = brew.conv(model, self.conv3, 'conv4', dim_in=48, dim_out=64, kernel=3)
	    # Image size: 3x20 -> 1x18
        self.conv5 = brew.conv(model, self.conv4, 'conv5', dim_in=64, dim_out=64, kernel=3)
	    # Flatten from 64 X 1 X 18 image to the "deep feature" vector
        self.deep_features = model.net.Reshape("conv5", ["deep_features", "conv5_old"], shape=[-1, 64*1*18])
        self.fc1 = brew.fc(model, 'deep_features', 'fc1', dim_in=64*1*18, dim_out=100, axis=1)
        self.fc2 = brew.fc(model, self.fc1, 'fc2', dim_in=100, dim_out=50, axis=1)
        self.fc3 = brew.fc(model, self.fc2, 'fc3', dim_in=50, dim_out=10, axis=1)
        self.prediction = brew.fc(model, self.fc3, 'prediction', dim_in=10, dim_out=self.output_dim, axis=1)
        # Create a copy of the current net. We will use it on the forward
        # pass where we don't need loss and backward operators
        self.forward_net = core.Net(model.net.Proto())
        #end forward pass. Add stuff needed for the backward pass and loss minimization
        target =  model.net.AddExternalInputs('target')
        self.squared_norms = model.net.SquaredL2Distance([self.prediction, target], 'squared_norms', axis = 1)
        #self.norms = model.net.Sqrt(squared_norms, 'l2_norms')
        # Loss is average across batch
        #self.loss = self.squared_norms.AveragedLoss([], ["loss"])
        self.loss = model.net.AveragedLoss(self.squared_norms, 'loss')
        model.AddGradientOperators([self.loss])
        # use build_sdg function to build an optimizer
        build_sgd(model,base_learning_rate=0.0000001,policy="step",stepsize=1,gamma=1.0)
        self.model = model  
    def add_training_operators(softmax, model, device_opts) :
        xent = model.LabelCrossEntropy([softmax, "label"], 'xent')
        loss = model.AveragedLoss(xent, "loss")
        brew.accuracy(model, [softmax, "label"], "accuracy")
        model.AddGradientOperators([loss])
        opt = optimizer.build_sgd(model, base_learning_rate=0.01, policy="step", stepsize=1, gamma=0.999) 
    def TrainModel(self):
        workspace.RunNetOnce(self.model.param_init_net)
        images, labels = self.read_data(self.input_folder,self.input_file,"raw_images")
        num_images = images.shape[0]
        possible_vals = np.linspace(0,num_images-1,num_images).astype(np.int32)
        np.random.seed()
        input = np.random.rand(self.batch_size, self.input_channels, self.input_height, self.input_width).astype(np.float32)
        target = np.random.rand(self.batch_size, self.output_dim).astype(np.float32)
        workspace.FeedBlob('input_blob', input)
        workspace.FeedBlob('target', target)
        CreateNetOnce(self.model.net)
        for n in range(1,self.num_training_iterations):
            np.random.shuffle(possible_vals)
            indices = possible_vals[0:self.batch_size]
            input = images[indices,:].astype(np.float32)
            input = np.divide(input, self.scale_factor)
            target = labels[indices,:].astype(np.float32)
            workspace.FeedBlob('input_blob', input)
            workspace.FeedBlob('target', target)
          #  CreateNetOnce(self.model.net)
            workspace.RunNet(self.model.net.Name())
            loss = workspace.FetchBlob(self.loss)
            print("loss", loss)
        init_pb, predictor_pb = Export(workspace, self.forward_net, self.model.params)   
        file_prefix, file_extension = self.input_file.split(".")
        text_file = open("init_net_" + file_prefix + ".pb", "wb")
        text_file.write(init_pb.SerializeToString())
        text_file.close()
        text_file = open("predict_net_" + file_prefix + ".pb","wb")
        text_file.write(predictor_pb.SerializeToString())
        text_file.close()
           # break
    def read_data(self, root_folder, annotations_file, images_folder):
        im_folder = os.path.join(root_folder, images_folder)
        ann_file = os.path.join(root_folder, annotations_file)
        f = open(ann_file, "r")
        # use readlines to read all lines in the file
        # The variable "lines" is a list containing all lines in the file
        lines = f.readlines()
        # close the file after reading the lines.
        f.close()
        rtn = np.random.rand(len(lines), self.input_channels, self.input_height, self.input_width).astype(np.float32)
        labels = np.random.rand(len(lines), self.output_dim).astype(np.float32)
        for i in range(len(lines)):
            line = lines[i]
            im_file, _, steering_angle = line.split(",")
            full_image_path = os.path.join(im_folder, im_file)
            img = cv2.imread(full_image_path, cv2.IMREAD_UNCHANGED)
            img_resized= cv2.resize(img,dsize=(self.input_width,self.input_height), interpolation = cv2.INTER_CUBIC)
            img_transposed = np.transpose(img_resized, (2, 0, 1))
            rtn[i]=img_transposed.astype(np.float32)
            labels[i][0]=100.0*float(steering_angle)
        labels = labels.astype(np.float32)
        return rtn, labels
@utils.debug
def main():
    parser = argparse.ArgumentParser(description="Caffe2: PilotNet Training")
    #  parser.add_argument("--train_data", type=str, default=None,
    #                      help="Path to training data in a text file format",
    #                      required=True)
    parser.add_argument("--gpu", type=int, default=-1,  help="If set, training is going to use the GPU specified")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch Size")
    parser.add_argument("--output_dim", type=int, default=1, help="Dimensionality of predicted control input")
    parser.add_argument("--num_iterations", type=int, default=10000, help="Number of iterations to train")
    parser.add_argument("--input_file", type=str, default="annotations.csv", help="Input file to pull annotations from")
    parser.add_argument("--input_folder", type=str, required=True, help="Input file to pull annotations from")
    parser.add_argument("--scale_factor", type=float, default=2.55, help="Scaling factor to divide all of the pixels in the training images by")
    args = parser.parse_args()
    last_time = datetime.now()
    progress = 0
    device = core.DeviceOption(caffe2_pb2.CUDA if args.gpu>=0 else caffe2_pb2.CPU, args.gpu)
    with core.DeviceScope(device):
        pilotnet = PilotNet(args)
        pilotnet.CreateModel()
        print("yay")
        pilotnet.TrainModel()

     #   init_pb, predictor_pb = Export(workspace, self.model.net, self.model.GetParams())   
    #    print(predictor_pb)
       # print(init_pb)
if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()



