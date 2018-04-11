from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python.net_builder import ops
from caffe2.python import core, workspace, model_helper, utils, brew
import urllib2
from caffe2.python.rnn_cell import LSTM
from caffe2.proto import caffe2_pb2

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
    device = core.DeviceOption(caffe2_pb2.CUDA , 0)
    with core.DeviceScope(device):
        SCALE_FACTOR=2.55
        INIT_NET = "init_net_test_file.pb"
        PREDICT_NET = "predict_net_test_file.pb"
        init_def = caffe2_pb2.NetDef()
        with open(INIT_NET, 'rb') as f:
            init_def.ParseFromString(f.read())
            init_def.device_option.CopyFrom(device)
        workspace.RunNetOnce(init_def.SerializeToString())
        print(init_def)
        net_def = caffe2_pb2.NetDef()
        with open(PREDICT_NET, 'rb') as f:
            net_def.ParseFromString(f.read())
            net_def.device_option.CopyFrom(device)
            workspace.CreateNet(net_def.SerializeToString())
        #`#run the net and return prediction
        print(net_def)
        img = cv2.imread("D:/test_data/slow_run_australia_track2/raw_images/raw_image_756.jpg", cv2.IMREAD_UNCHANGED)
        img_resized= cv2.resize(img,dsize=(200,66), interpolation = cv2.INTER_CUBIC)
        img_transposed = np.transpose(img_resized, (2, 0, 1)).astype(np.float32)
        input = np.random.rand(1,3,66,200).astype(np.float32)
        input[0] = img_transposed
        input = np.divide(input, SCALE_FACTOR)
        print(input)

        workspace.FeedBlob('input_blob', input, device_option=device)
        workspace.RunNet("PilotNet_1")
        workspace.FetchBlob("prediction")
if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()
