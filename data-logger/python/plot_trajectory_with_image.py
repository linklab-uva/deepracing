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
parser.add_argument("label", help="Label Index to Use",  type=int)
args = parser.parse_args()
dbpath = args.dbpath
labelindex = args.label
#/home/ttw2xk/f1_data/madhur_head2head_3/pose_sequence_labels
labelpath = os.path.join(dbpath,"pose_sequence_labels","image_%d_sequence_label.json" %(labelindex))
labelimagepath = os.path.join(dbpath,"images","image_%d.jpg" %(labelindex))

label = PoseSequenceLabel_pb2.PoseSequenceLabel()
with open(labelpath,'r') as f:
    pbjson.Parse(f.read(), label)

timeslabel = np.array([p.session_time for p in label.subsequent_poses])

labelimage = tf.to_tensor(imutils.readImage(labelimagepath))

labelimage = labelimage[:,32:,:]

labelimage = tf.to_tensor(tf.resize(tf.to_pil_image(labelimage), int(0.5*labelimage.shape[1])))

labelimage_np = (255.0*labelimage).numpy().copy().astype(np.uint8).transpose(1,2,0)

labelposes = label.subsequent_poses

labelpositions = [p.translation for p in labelposes]

labelpositionsnp = np.array([np.array((t.x, t.y, t.z)) for t in labelpositions])



label_x = labelpositionsnp[:,0]
label_z = labelpositionsnp[:,2]

fig, (ax1, ax2) = plt.subplots(2)
#fig.suptitle('An example of head-to-head racing data')
ax1.set_title("Head-To-Head Image")
ax1.imshow(labelimage_np)
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
ax2.set_title("Trajectory Label")
ax2.set_xlabel("X distance (meters)")
ax2.set_ylabel("Z distance (meters)")
ax2.set_xlim((-25,25))
ax2.plot(label_x, label_z)
fig.tight_layout()
os.makedirs(os.path.join(dbpath,"figures"),exist_ok=True)
plt.savefig(os.path.join(dbpath,"figures","image_%d_withlabel.eps"%(labelindex)))
plt.savefig(os.path.join(dbpath,"figures","image_%d_withlabel.svg"%(labelindex)))
plt.savefig(os.path.join(dbpath,"figures","image_%d_withlabel.png"%(labelindex)))
plt.show()