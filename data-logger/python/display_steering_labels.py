import numpy as np
import numpy.linalg as la
import scipy
import scipy.stats
import skimage
import PIL
from PIL import Image as PILImage
import TimestampedPacketMotionData_pb2
import PoseSequenceLabel_pb2
import TimestampedImage_pb2
import Vector3dStamped_pb2
import argparse
import os
import google.protobuf.json_format
import Pose3d_pb2
import cv2
import bisect
import FrameId_pb2
import scipy.interpolate
import deepracing.pose_utils
from deepracing.protobuf_utils import getAllImageFilePackets, getAllMotionPackets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import deepracing.backend.LabelBackends as LB
from tqdm import tqdm as tqdm

def imageDataKey(data):
    return data.timestamp
parser = argparse.ArgumentParser()
parser.add_argument("db_path", help="Path to db",  type=str)
args = parser.parse_args()
db_path = args.db_path
image_path = os.path.join(db_path,"images")
key_file = os.path.join(db_path,"controloutputkeys.txt")

steering_lmdb_path = os.path.join(db_path,"steering_labels","lmdb")
label_backend = LB.ControlLabelLMDBWrapper()
label_backend.readDatabase(steering_lmdb_path)
with open(key_file,"r") as f:
    keystrings = f.readlines()
    keys = [keystring.replace('\n','') for keystring in keystrings]
thickness = 9
color = (0, 255, 0)
cv2.namedWindow("image",cv2.WINDOW_AUTOSIZE)
for key in tqdm(keys):
    im = cv2.imread(os.path.join(image_path,key+".jpg"))
    label = label_backend.getControlLabel(key)
    print(label)
    angle = -1.0*label.label.steering*np.pi/2 + np.pi/2
    pt1 = (np.flip(np.array(im.shape[0:2]))/2).astype(np.int32)
    print(pt1)
    # R = np.array([[np.cos(np.pi,),    -np.sin(np.pi)],
    #               [np.sin(np.pi),     np.cos(np.pi)]
    #               ])
    R = np.eye(2)
    v = np.matmul(R.transpose(),np.array((np.cos(angle), -np.sin(angle))))
    pt2 = (pt1 + v*50).astype(np.int32)
    imwithline = cv2.arrowedLine(im, tuple(pt1.tolist()), tuple(pt2.tolist()), color, thickness)
    cv2.imshow("image",imwithline)
    cv2.waitKey(0)
    
   # print(key)