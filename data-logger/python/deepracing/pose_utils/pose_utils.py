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

def labelPacketToNumpy(label_tag):
    #print(label_tag.subsequent_poses)
    positions = np.array([np.array((pose.translation.x,pose.translation.y, pose.translation.z)) for pose in label_tag.subsequent_poses])
    quats = np.array([np.array((pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w)) for pose in label_tag.subsequent_poses])
    linear_velocities = np.array([np.array((vel.vector.x, vel.vector.y, vel.vector.z)) for vel in label_tag.subsequent_linear_velocities])
    angular_velocities = np.array([np.array((vel.vector.x, vel.vector.y, vel.vector.z)) for vel in label_tag.subsequent_angular_velocities])
    return positions, quats, linear_velocities, angular_velocities
def getAllSequenceLabelPackets(label_packet_folder: str, use_json: bool = False):
   label_packets = []
   if use_json:
      filepaths = [os.path.join(label_packet_folder, f) for f in os.listdir(label_packet_folder) if os.path.isfile(os.path.join(label_packet_folder, f)) and str.lower(os.path.splitext(f)[1])==".json"]
      jsonstrings = [(open(path, 'r')).read() for path in filepaths]
      for jsonstring in jsonstrings:
         data = PoseSequenceLabel_pb2.PoseSequenceLabel()
         google.protobuf.json_format.Parse(jsonstring, data)
         label_packets.append(data)
   else:
      filepaths = [os.path.join(label_packet_folder, f) for f in os.listdir(label_packet_folder) if os.path.isfile(os.path.join(label_packet_folder, f)) and str.lower(os.path.splitext(f)[1])==".pb"]
      for filepath in filepaths:
         try:
            data = PoseSequenceLabel_pb2.PoseSequenceLabel()
            f = open(filepath,'rb')
            data.ParseFromString(f.read())
            f.close()
            label_packets.append(data)
         except Exception as e:
            f.close()
            print(str(e))
            print("Could not read binary file %s." %(filepath))
            continue
   return label_packets
def getAllMotionPackets(motion_data_folder: str, use_json: bool):
   motion_packets = []
   if use_json:
      filepaths = [os.path.join(motion_data_folder, f) for f in os.listdir(motion_data_folder) if os.path.isfile(os.path.join(motion_data_folder, f)) and str.lower(os.path.splitext(f)[1])==".json"]
      jsonstrings = [(open(path, 'r')).read() for path in filepaths]
      for jsonstring in jsonstrings:
         data = TimestampedPacketMotionData_pb2.TimestampedPacketMotionData()
         google.protobuf.json_format.Parse(jsonstring, data)
         motion_packets.append(data)
   else:
      filepaths = [os.path.join(motion_data_folder, f) for f in os.listdir(motion_data_folder) if os.path.isfile(os.path.join(motion_data_folder, f)) and str.lower(os.path.splitext(f)[1])==".pb"]
      for filepath in filepaths:
         try:
            data = TimestampedPacketMotionData_pb2.TimestampedPacketMotionData()
            f = open(filepath,'rb')
            data.ParseFromString(f.read())
            f.close()
            motion_packets.append(data)
         except:
            f.close()
            print("Could not read udp file %s." %(filepath))
            continue
   return motion_packets
def getAllImageFilePackets(image_data_folder: str, use_json: bool):
   image_packets = []
   if use_json:
      filepaths = [os.path.join(image_data_folder, f) for f in os.listdir(image_data_folder) if os.path.isfile(os.path.join(image_data_folder, f)) and str.lower(os.path.splitext(f)[1])==".json"]
      jsonstrings = [(open(path, 'r')).read() for path in filepaths]
      for jsonstring in tqdm(jsonstrings):
         data = TimestampedImage_pb2.TimestampedImage()
         google.protobuf.json_format.Parse(jsonstring, data)
         image_packets.append(data)
   else:
      filepaths = [os.path.join(image_data_folder, f) for f in os.listdir(image_data_folder) if os.path.isfile(os.path.join(image_data_folder, f)) and str.lower(os.path.splitext(f)[1])==".pb"]
      for filepath in tqdm(filepaths):
         try:
            data = TimestampedImage_pb2.TimestampedImage()
            f = open(filepath,'rb')
            data.ParseFromString(f.read())
            f.close()
            image_packets.append(data)
         except:
            f.close()
            print("Could not read image data file %s." %(filepath))
            continue
   return image_packets

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

def extractAngularVelocity(packet):
    angular_velocity = np.array((packet.m_angularVelocityX, packet.m_angularVelocityY, packet.m_angularVelocityZ), np.float64)
    return angular_velocity
    
def extractVelocity(packet, car_index = 0):
    motion_data = packet.m_carMotionData[car_index]
    velocity = np.array((motion_data.m_worldVelocityX, motion_data.m_worldVelocityY, motion_data.m_worldVelocityZ), np.float64)
    return velocity
    
def extractPose(packet, car_index = 0):
    motion_data = packet.m_carMotionData[car_index]
    rightvector = np.array((motion_data.m_worldRightDirX, motion_data.m_worldRightDirY, motion_data.m_worldRightDirZ), dtype=np.float64)
    rightvector = rightvector/la.norm(rightvector)
    forwardvector = np.array((motion_data.m_worldForwardDirX, motion_data.m_worldForwardDirY, motion_data.m_worldForwardDirZ), dtype=np.float64)
    forwardvector = forwardvector/la.norm(forwardvector)
    upvector = np.cross(rightvector,forwardvector)
    upvector = upvector/la.norm(upvector)
    rotationmat = np.vstack((-rightvector,upvector,forwardvector)).transpose()
    position = np.array((motion_data.m_worldPositionX, motion_data.m_worldPositionY, motion_data.m_worldPositionZ), dtype=np.float64)
    quat = Rot.from_dcm(rotationmat).as_quat()
    return position, quat 
