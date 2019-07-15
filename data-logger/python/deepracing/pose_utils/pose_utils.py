import TimestampedPacketMotionData_pb2
import PoseSequenceLabel_pb2
import TimestampedImage_pb2
import Vector3dStamped_pb2
import os
import numpy as np
import quaternion
import numpy.linalg as la
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
      for jsonstring in jsonstrings:
         data = TimestampedImage_pb2.TimestampedImage()
         google.protobuf.json_format.Parse(jsonstring, data)
         image_packets.append(data)
   else:
      filepaths = [os.path.join(image_data_folder, f) for f in os.listdir(image_data_folder) if os.path.isfile(os.path.join(image_data_folder, f)) and str.lower(os.path.splitext(f)[1])==".pb"]
      for filepath in filepaths:
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
def fromHomogenousTransformArray(transform):
    pos = transform[:,0:3,3].copy()
    quat = quaternion.from_rotation_matrix(transform[:,0:3,0:3])
    return pos,quat
def fromHomogenousTransform(transform):
    pos = transform[0:3,3].copy()
    quat = quaternion.from_rotation_matrix(transform[0:3,0:3])
    return pos,quat
def toHomogenousTransformArray(position, quat):
    rtn = np.array([np.eye(4) for i in range(position.shape[0])])
    rtn[:,0:3,3] = position.copy()
    rtn[:,0:3,0:3] = np.array([quaternion.as_rotation_matrix(quat[i]) for i in range(quat.shape[0])])
    return rtn
def toHomogenousTransform(position, quat):
    rtn = np.eye(4)
    rtn[0:3,3] = position.copy()
    rtn[0:3,0:3] = quaternion.as_rotation_matrix(quat)
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
   # print(motion_data)
    rightvector = np.array((motion_data.m_worldRightDirX, motion_data.m_worldRightDirY, motion_data.m_worldRightDirZ), dtype=np.float64)
   # rightvector = rightvector/32767.0
    rightvector = rightvector/la.norm(rightvector)
    forwardvector = np.array((motion_data.m_worldForwardDirX, motion_data.m_worldForwardDirY, motion_data.m_worldForwardDirZ), dtype=np.float64)
  #  forwardvector = forwardvector/32767.0
    forwardvector = forwardvector/la.norm(forwardvector)
    upvector = np.cross(rightvector,forwardvector)
    upvector = upvector/la.norm(upvector)
	#rotationMat.col(0) = -right;
	#rotationMat.col(1) = up;
	#rotationMat.col(2) = forward;
    rotationmat = np.vstack((-rightvector,upvector,forwardvector)).transpose()
    #print(rotationmat)
    position = np.array((motion_data.m_worldPositionX, motion_data.m_worldPositionY, motion_data.m_worldPositionZ), dtype=np.float64)
    quat = quaternion.from_rotation_matrix(rotationmat)
    return position, quat 
