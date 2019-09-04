import numpy as np
import numpy.linalg as la
import quaternion
import scipy
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
import deepracing.backend
from deepracing.pose_utils import getAllImageFilePackets, getAllMotionPackets
from deepracing.protobuf_utils import getAllSessionPackets
from tqdm import tqdm as tqdm

def LabelPacketSortKey(packet):
    return packet.car_pose.session_time

parser = argparse.ArgumentParser()
parser.add_argument("db_path", help="Path to root directory of pose sequence label LMDB",  type=str)
parser.add_argument("key_file", help="Path to output list of db keys sorted in order by session timestamp",  type=str)
args = parser.parse_args()
label_db = args.db_path
key_file = args.key_file

label_wrapper = deepracing.backend.PoseSequenceLabelLMDBWrapper()
label_wrapper.readDatabase(label_db, max_spare_txns=1)
db_keys = label_wrapper.getKeys()
label_pb_tags = []
print("Loading database labels.")
for i,key in tqdm(enumerate(db_keys), total=len(db_keys)):
    #print(key)
    label_pb_tags.append(label_wrapper.getPoseSequenceLabel(key))
    if(not (label_pb_tags[-1].image_tag.image_file == db_keys[i]+".jpg")):
        raise AttributeError("Mismatch between database key: %s and associated image file: %s" %(db_keys[i], label_pb_tags.image_tag.image_file))
label_pb_tags = sorted(label_pb_tags, key=LabelPacketSortKey)
sorted_keys = [os.path.splitext(packet.image_tag.image_file)[0] for packet in label_pb_tags]
with open(key_file, 'w') as filehandle:
    filehandle.writelines("%s\n" % key for key in sorted_keys)