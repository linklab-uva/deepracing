import numpy as np
import numpy.linalg as la
import quaternion
import scipy
import skimage
import PIL
from PIL import Image as PILImage
import TimestampedPacketMotionData_pb2
import argparse
import os
import google.protobuf.json_format
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import TimestampedImage_pb2
import Pose3d_pb2
import cv2
import PoseSequenceLabel_pb2
import bisect
import FrameId_pb2
import Vector3dStamped_pb2
import scipy.interpolate

v1 = np.random.rand(3)
v1 = v1/la.norm(v1)

vrand = np.random.rand(3)
vrand = vrand/la.norm(vrand)

v2 = np.cross(v1,vrand)
v2 = v2/la.norm(v2)

v3 =  np.cross(v1,v2)
v3 = v3/la.norm(v3)

rotationmat = np.vstack( (v1, v2, v3) ).transpose()
print(rotationmat)
quat = quaternion.from_rotation_matrix(rotationmat)
quatcopy = quaternion.quaternion()
quatcopy.x = quat.x
quatcopy.y = quat.y
quatcopy.z = quat.z
quatcopy.w = quat.w
rotationmatout = quaternion.as_rotation_matrix(quatcopy).copy()
print(rotationmat)

print(la.norm(rotationmat - rotationmatout))
