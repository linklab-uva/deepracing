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
from deepracing.pose_utils import getAllImageFilePackets, getAllMotionPackets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def imageDataKey(data):
    return data.timestamp
parser = argparse.ArgumentParser()
parser.add_argument("image_path", help="Path to image folder",  type=str)
parser.add_argument("--json", help="Assume dataset files are in JSON rather than binary .pb files.",  action="store_true")
args = parser.parse_args()
image_folder = args.image_path
image_tags = deepracing.pose_utils.getAllImageFilePackets(image_folder, args.json)
image_tags = sorted(image_tags,key=imageDataKey)
print(image_tags)
indices = np.array([float(i) for i in range(len(image_tags))])
timestamps = np.array([tag.timestamp/1000.0 for tag in image_tags])
fig = plt.figure("Image Index vs OS Time")
plt.plot(timestamps, indices, label='indices versus timestamps')
fig.legend()
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(timestamps, indices)
print("Average framerate: %f" %(slope))
#plt.plot( timestamps, slope*timestamps + intercept, label='fitted line' )
plt.show()

