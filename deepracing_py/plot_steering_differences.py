import google.protobuf.json_format as pbjson
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import LabeledImage_pb2
import argparse
import deepracing.imutils as imutils
import torchvision
import torchvision.utils as tvutils
import torchvision.transforms.functional as tf
import torch
def loadLabelFiles(labeldir,imagefilename):

    labelstart = int(str.split(imagefilename,"_")[1])
    #image_2_control_label.json
    rtn = []
    for i in range(labelstart, labelstart + 60):
        label = LabeledImage_pb2.LabeledImage()
        fp = os.path.join(labeldir,"image_%d_control_label.json" % (i,))
        with open(fp,'r') as f:
            pbjson.Parse(f.read(), label)
        rtn.append(label)
    return rtn
def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
#D:\f1_training_data\trent_solo_4\pose_sequence_labels\image_36_sequence_label.json D:\f1_training_data\trent_solo\pose_sequence_labels\image_65_sequence_label.json
# are strong candidates
parser = argparse.ArgumentParser()

parser.add_argument("label1", help="First image file",  type=str)
parser.add_argument("label2", help="Second image file",  type=str)
args = parser.parse_args()

image1path = args.label1
image2path = args.label2

image1 = tf.to_tensor(imutils.readImage(image1path))
image2 = tf.to_tensor(imutils.readImage(image2path))



label1steeringdir = os.path.join(os.path.dirname(os.path.dirname(image1path)),"steering_labels")
label2steeringdir = os.path.join(os.path.dirname(os.path.dirname(image2path)),"steering_labels")


image1name = os.path.splitext(os.path.basename(image1path))[0]
image2name = os.path.splitext(os.path.basename(image2path))[0]



labels1 = loadLabelFiles(label1steeringdir, image1name)
labels2 = loadLabelFiles(label2steeringdir, image2name)

steering1 = [l.label.steering for l in labels1]
steering2 = [l.label.steering for l in labels2]

image1 = image1[:,32:,:]
image2 = image2[:,32:,:]

image1 = tf.to_tensor(tf.resize(tf.to_pil_image(image1), int(0.5*image1.shape[1])))

image2 = tf.to_tensor(tf.resize(tf.to_pil_image(image2), image1.shape[1:]))


fig1 = plt.figure()
t = np.linspace(0,1.42,60)
plt.plot(t, steering1)
plt.plot(t, steering2)
plt.xlabel("Time (Seconds)")
plt.ylabel("Normalized Steering Angles [-1, 1]")

fig2 = plt.figure()
imagegrid = tvutils.make_grid([image1, image2], nrow=1)
show(imagegrid)
plt.show()