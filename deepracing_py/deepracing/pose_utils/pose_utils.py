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
import deepracing
import scipy, scipy.stats

def registerImagesToMotiondata(motion_packets, image_tags):
    motion_packets = sorted(motion_packets, key=deepracing.timestampedUdpPacketKey)
    motion_packet_session_times = np.array([packet.udp_packet.m_header.m_sessionTime for packet in motion_packets])
    motion_packet_system_times = np.array([packet.timestamp/1000.0 for packet in motion_packets])

    image_tags = [tag for tag in image_tags if tag.timestamp/1000.0<motion_packet_system_times[-1]]
    image_tags = sorted(image_tags, key = deepracing.imageDataKey)
    image_system_timestamps = np.array([data.timestamp/1000.0 for data in image_tags])

    #find first packet with a system timestamp >= the system timestamp of the first image
    Imin = motion_packet_system_times>=image_system_timestamps[0]
    firstIndex = np.argmax(Imin)
    motion_packets = motion_packets[firstIndex:]
    motion_packets = sorted(motion_packets, key=deepracing.timestampedUdpPacketKey)
    
    #filter out duplicate packets (which happens for occasionally for some reason)
    motion_packet_session_times = np.array([packet.udp_packet.m_header.m_sessionTime for packet in motion_packets])
    unique_session_times, unique_session_time_indices = np.unique(motion_packet_session_times, return_index=True)
    motion_packets = [motion_packets[i] for i in unique_session_time_indices]
    motion_packets = sorted(motion_packets, key=deepracing.timestampedUdpPacketKey)

    motion_packet_session_times = np.array([packet.udp_packet.m_header.m_sessionTime for packet in motion_packets])
    motion_packet_system_times = np.array([packet.timestamp/1000.0 for packet in motion_packets])

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(motion_packet_system_times, motion_packet_session_times)
    print("Slope and intercept of session time vs system time: [%f,%f]" %(slope, intercept))
    print( "r value of session time vs system time: %f" % ( r_value ) )
    print( "r^2 value of session time vs system time: %f" % ( r_value**2 ) )
    image_session_timestamps = slope*image_system_timestamps + intercept
    print("Range of image session times before clipping: [%f,%f]" %(image_session_timestamps[0], image_session_timestamps[-1]))
    if not len(image_tags) == len(image_session_timestamps):
        raise ValueError("Different number of image tags (%d) than image session timestamps (%d)" % (len(image_tags), len(image_session_timestamps)) )

    #clip the image tags to be within the range of the known session times
    Iclip = (image_session_timestamps>np.min(motion_packet_session_times)) * (image_session_timestamps<np.max(motion_packet_session_times))
    image_tags = [image_tags[i] for i in range(len(image_tags)) if Iclip[i]]
    image_session_timestamps = image_session_timestamps[Iclip]
    print("Range of image session times after clipping: [%f,%f]" %(image_session_timestamps[0], image_session_timestamps[-1]))
    if not len(image_tags) == len(image_session_timestamps):
        raise ValueError("Different number of image tags (%d) than image session timestamps (%d)" % (len(image_tags), len(image_session_timestamps)) )
    return image_tags, image_session_timestamps, motion_packets, slope, intercept

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
def toHomogenousTransform(position, quat):
    rtn = np.eye(4)
    rtn[0:3,3] = position.copy()
    rtn[0:3,0:3] = Rot.from_quat(quat).as_dcm()
    return rtn
def toHomogenousTransformArray(positions, quats):
    length = positions.shape[0]
    assert(quats.shape[0]==length)
    rtn = np.array([toHomogenousTransform(positions[i], quats[i]) for i in range(length)])
    return rtn
def interpolateVectors(p1, t1, p2, t2, t_interp):
    tau = (t_interp - t1)/(t2 - t1)
    rtn = (1-tau)*p1 + tau*p2
    return rtn
