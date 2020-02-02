import TimestampedPacketMotionData_pb2
import PoseSequenceLabel_pb2
import TimestampedImage_pb2
import Vector3dStamped_pb2
import PoseSequenceLabel_pb2
import os
import numpy as np
import numpy.linalg as la
import google.protobuf.json_format
from tqdm import tqdm as tqdm
from scipy import interpolate
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import RotationSpline as RotSpline
def randomTransform(pscale=1.0):
   T = np.eye(4)
   T[0:3,0:3] = Rot.random().as_dcm()
   T[0:3,3]=pscale*(np.random.rand(3)-np.random.rand(3))
   return T
def randomQuaternion():
   return Rot.random().as_quat()
def toLocalCoordinatesVector(coordinate_system, vectors):
    pose_mat = toHomogenousTransform( coordinate_system[0] , coordinate_system[1] )
    pose_mat_inv = inverseTransform( pose_mat )
    return np.transpose(np.matmul(pose_mat_inv[0:3,0:3],np.transpose(vectors).copy()))
def toLocalCoordinatesPose(coordinate_system, positions, quaternions):
    assert(positions.shape[0] == quaternions.shape[0])
    pose_mat = toHomogenousTransform( coordinate_system[0] , coordinate_system[1] )
    pose_mat_inv = inverseTransform( pose_mat )
    poses_to_transform = toHomogenousTransformArray(positions, quaternions)
    poses_transformed = np.matmul(pose_mat_inv,poses_to_transform)
    positions, quats = fromHomogenousTransformArray(poses_transformed)
    return positions, quats
def inverseTransform(transform):
   rtn = transform.copy()
   rtn[0:3,0:3] = np.transpose(rtn[0:3,0:3])
   rtn[0:3,3] = np.matmul(rtn[0:3,0:3], -rtn[0:3,3])
   return rtn
   #return la.inv(transform)
def fromHomogenousTransformArray(transforms):
    positions = transforms[:,0:3,3].copy()
    quats = Rot.from_dcm(transforms[:,0:3,0:3]).as_quat()
    return positions, quats
def fromHomogenousTransform(transform):
    pos = transform[0:3,3].copy()
    quat = Rot.from_dcm(transform[0:3,0:3]).as_quat()
    return pos,quat
def toHomogenousTransformArray(positions, quats):
    length = positions.shape[0]
    assert(quats.shape[0]==length)
    rtn = np.array([np.eye(4) for i in range(length)])
    rtn[:,0:3,3] = positions.copy()
    rtn[:,0:3,0:3] = Rot.from_quat(quats).as_dcm()
    return rtn
def toHomogenousTransform(position, quat):
    rtn = np.eye(4)
    rtn[0:3,3] = position.copy()
    rtn[0:3,0:3] = Rot.from_quat(quat).as_dcm()
    return rtn
def interpolateVectors(p1, t1, p2, t2, t_interp):
    tau = (t_interp - t1)/(t2 - t1)
    rtn = (1-tau)*p1 + tau*p2
    return rtn
