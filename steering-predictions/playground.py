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
from pyf1_datalogger import ScreenVideoCapture as svc
import cv2 as cv
import numpy as np
import argparse
def main():
    parser = argparse.ArgumentParser(description="Deepf1 playground")
    parser.add_argument("--app", type=str, default="",  help="Application to cature. Captures the desktop if not specified")
    parser.add_argument("--x", type=int, default=0,  help="X coordinate of uppper-left corner of box to capture")
    parser.add_argument("--y", type=int, default=0,  help="Y coordinate of uppper-left corner of box to capture")
    parser.add_argument("--w", type=int, default=50,  help="Width of box to capture")
    parser.add_argument("--h", type=int, default=50,  help="Height of box to capture")
    with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
        SCALE_FACTOR=2.55
        INIT_NET = "init_net_quadratic_interpolation.pb"
        PREDICT_NET = "predict_net_quadratic_interpolation.pb"
        init_def = caffe2_pb2.NetDef()
        with open(INIT_NET, 'rb') as f:
            init_def.ParseFromString(f.read())
            workspace.RunNetOnce(init_def.SerializeToString())
        #print(init_def)
        net_def = caffe2_pb2.NetDef()
        with open(PREDICT_NET, 'rb') as f:
            net_def.ParseFromString(f.read())
            workspace.CreateNet(net_def.SerializeToString())
        args = parser.parse_args() 
        app = args.app
        x = args.x
        y = args.y
        w = args.w
        h = args.h
        capture = svc()
        capture.open(app, x,y,w,h)
        im = capture.read()
        print(im)
        print(type(im))
        print(im.shape)
        #im = im.astype(np.int32)
        cv.imshow('image',im)
        cv.waitKey(0)
        cv.destroyAllWindows()
        img_resized= cv.resize(im,dsize=(200,66), interpolation = cv.INTER_CUBIC)
        img_noalpha = img_resized[:,:,0:1:2]
        img_transposed = np.transpose(img_noalpha, (2, 0, 1)).astype(np.float32)
        input = np.random.rand(1,3,66,200).astype(np.float32)
        input[0] = img_transposed
        input_scaled = np.divide(input, SCALE_FACTOR).astype(np.float32)
        print("Scaled input shape: ", input_scaled.shape)
        workspace.FeedBlob('input_blob', input_scaled)
        cv.imshow('image',img_resized)
        cv.waitKey(0)
        cv.destroyAllWindows()
        workspace.RunNet("PilotNet_1")
       # pred = workspace.FetchBlob("prediction")

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()

