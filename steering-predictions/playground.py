from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import cv2
from caffe2.python.net_builder import ops
from caffe2.python import core, workspace, model_helper, utils, brew
from caffe2.python.rnn_cell import LSTM
from caffe2.proto import caffe2_pb2
from caffe2.python.optimizer import build_sgd
from caffe2.python.predictor.mobile_exporter import Export, add_tensor
from caffe2.python.predictor.predictor_exporter import get_predictor_exporter_helper, PredictorExportMeta
from pyf1_datalogger import ScreenVideoCapture as svc
import img_utils.utils as imutils
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Deepf1 playground")
    parser.add_argument("--app", type=str, default="",  help="Application to cature. Captures the desktop if not specified")
    parser.add_argument("--x", type=int, default=0,  help="X coordinate of uppper-left corner of box to capture")
    parser.add_argument("--y", type=int, default=0,  help="Y coordinate of uppper-left corner of box to capture")
    background = cv2.imread('test_img.jpg', cv2.IMREAD_UNCHANGED)
    wheel = cv2.imread('steering_wheel.png', cv2.IMREAD_UNCHANGED)  
    cv2.namedWindow('Display image')          ## create window for display
    imutils.overlay_image(background,wheel,0,0)
    cv2.imshow('Display image', background)          ## Show image in the window
    cv2.waitKey(0)

    

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()

