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
import cv2
import argparse
import logging
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
    def __init__(self):
        self.batch_size = 10
        self.output_dim = 1
        self.input_channels = 3
        self.input_height = 66
        self.input_width = 200
    def CreateModel(self):
        log.debug("Start training")
        model = model_helper.ModelHelper(name="PilotNet")
        input_blob, target =  model.net.AddExternalInputs( 'input_blob', 'target', )
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
        self.conv4 = brew.conv(model, self.conv3, 'conv4', dim_in=48, dim_out=64, kernel=3, stride=2)
	    # Image size: 3x20 -> 1x18
        self.conv5 = brew.conv(model, self.conv4, 'conv5', dim_in=64, dim_out=64, kernel=3, stride=2)
	    # Flatten from 64*1*18 image length to the "deep feature" vector
        self.fc1 = brew.fc(model, self.conv5, 'fc1', dim_in=64*1*18, dim_out=100)
        self.fc2 = brew.fc(model, self.fc1, 'fc2', dim_in=100, dim_out=50)
        self.fc3 = brew.fc(model, self.fc2, 'fc3', dim_in=50, dim_out=10)
        self.prediction = brew.fc(model, self.fc3, 'prediction', dim_in=10, dim_out=self.output_dim)
        # Create a copy of the current net. We will use it on the forward
        # pass where we don't need loss and backward operators
        self.forward_net = core.Net(model.net.Proto())
        self.squared_norms = model.net.SquaredL2Distance([self.prediction, target], 'l2_norms')
        # Loss is average both across batch and through time
        # Thats why the learning rate below is multiplied by self.seq_length
        self.loss = model.net.AveragedLoss(self.squared_norms, 'loss')
        model.AddGradientOperators([self.loss])
        # use build_sdg function to build an optimizer
        build_sgd(model,base_learning_rate=0.1,policy="step",stepsize=1,gamma=0.9999)
        self.model = model
@utils.debug
def main():
    parser = argparse.ArgumentParser(description="Caffe2: PilotNet Training")
    #  parser.add_argument("--train_data", type=str, default=None,
    #                      help="Path to training data in a text file format",
    #                      required=True)
    parser.add_argument("--gpu", action="store_true",  help="If set, training is going to use GPU 0")
    args = parser.parse_args()
    last_time = datetime.now()
    progress = 0
    device = core.DeviceOption(caffe2_pb2.CUDA if args.gpu else caffe2_pb2.CPU, 0)
    with core.DeviceScope(device):
        model = PilotNet()
        model.CreateModel()
        print("yay")

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()



