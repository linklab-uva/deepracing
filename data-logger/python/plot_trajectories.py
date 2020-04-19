import google.protobuf.json_format as pbjson
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import PoseSequenceLabel_pb2
import argparse
import deepracing.imutils as imutils
import torchvision
import torchvision.utils as tvutils
import torchvision.transforms.functional as tf
import torch

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
#D:\f1_training_data\trent_solo_4\pose_sequence_labels\image_36_sequence_label.json D:\f1_training_data\trent_solo\pose_sequence_labels\image_65_sequence_label.json
# are strong candidates
parser = argparse.ArgumentParser()
parser.add_argument("dbpath", help="main database path",  type=str)
parser.add_argument("label1", help="First label index",  type=int)
parser.add_argument("label2", help="Second label index",  type=int)
args = parser.parse_args()
dbpath = args.dbpath
label1 = args.label1
label2 = args.label2
#/home/ttw2xk/f1_data/madhur_head2head_3/pose_sequence_labels
label1path = os.path.join(dbpath,"pose_sequence_labels","image_%d_sequence_label.json" %(label1))
label2path = os.path.join(dbpath,"pose_sequence_labels","image_%d_sequence_label.json" %(label2))
label1imagepath = os.path.join(dbpath,"images","image_%d.jpg" %(label2))
label2imagepath = os.path.join(dbpath,"images","image_%d.jpg" %(label2))

label1 = PoseSequenceLabel_pb2.PoseSequenceLabel()
label2 = PoseSequenceLabel_pb2.PoseSequenceLabel()
with open(label1path,'r') as f:
    pbjson.Parse(f.read(), label1)
with open(label2path,'r') as f:
    pbjson.Parse(f.read(), label2)

timeslabel1 = np.array([p.session_time for p in label1.subsequent_poses])
timeslabel2 = np.array([p.session_time for p in label2.subsequent_poses]) 

label1image = tf.to_tensor(imutils.readImage(label1imagepath))
label2image = tf.to_tensor(imutils.readImage(label2imagepath))

label1image = label1image[:,32:,:]
label2image = label2image[:,32:,:]

label1image = tf.to_tensor(tf.resize(tf.to_pil_image(label1image), int(0.5*label1image.shape[1])))

label2image = tf.to_tensor(tf.resize(tf.to_pil_image(label2image), label1image.shape[1:]))



label1poses = label1.subsequent_poses
label2poses = label2.subsequent_poses

label1positions = [p.translation for p in label1poses]
label2positions = [p.translation for p in label2poses]

label1positionsnp = np.array([np.array((t.x, t.y, t.z)) for t in label1positions])
label2positionsnp = np.array([np.array((t.x, t.y, t.z)) for t in label2positions])



print(label1positionsnp)
print(label2positionsnp)



diff = label1positionsnp - label2positionsnp
diffsquare = np.square(diff)
squared_distances = np.sum(diffsquare,axis=1)
distances=np.sqrt(squared_distances)


label1_x = label1positionsnp[:,0]
label2_x = label2positionsnp[:,0]

label1_z = label1positionsnp[:,2]
label2_z = label2positionsnp[:,2]

print(label1_x)
print(label2_x)

fig1 = plt.figure()
plt.plot(-label1_x, label1_z)
plt.plot(-label2_x, label2_z)
plt.xlabel("X position in meters (Ego Vehicle Coordinates)")
plt.ylabel("Z position in meters (Ego Vehicle Coordinates)")


fig2 = plt.figure()
t = timeslabel1 - timeslabel1[0]
plt.plot(t, distances)
plt.xlabel("Time (Seconds)")
plt.ylabel("Distance between trajectories (meters)")

fig3 = plt.figure()
imagegrid = tvutils.make_grid([label1image, label2image], nrow=1)
meandistance = np.mean(distances)
maxdistance = np.max(distances)

print("Mean Distance: %f. Max Distance: %f" % (meandistance, maxdistance))
show(imagegrid)
plt.show()