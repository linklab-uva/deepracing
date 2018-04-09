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
    init_net_file = open("D:/deepracing/steering-predictions/init_net_test_file.pb", "rb")
    init_net_string = init_net_file.read()
    init_net_file.close()
    predict_net_file = open("D:/deepracing/steering-predictions/predict_net_test_file.pb", "rb")
    predict_net_string = predict_net_file.read()
    predict_net_file.close()
    device_opts=core.DeviceOption(caffe2_pb2.CUDA, 0)
    init_net = caffe2_pb2.NetDef()
    init_net.device_option.CopyFrom(device_opts)
    init_net.ParseFromString(init_net_string)
    predict_net= caffe2_pb2.NetDef()
    predict_net.device_option.CopyFrom(device_opts)
    predict_net.ParseFromString(predict_net_string)
    p = workspace.Predictor(init_net_string, predict_net_string)
    print(predict_net_string)
    print(p)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()
