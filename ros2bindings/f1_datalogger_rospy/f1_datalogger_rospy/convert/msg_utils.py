import f1_datalogger_msgs.msg as f1dmsgs # BezierCurve, TimestampedPacketMotionData, PacketMotionData, CarMotionData, PacketHeader
import geometry_msgs.msg as geo_msgs#  Point, PointStamped, Vector3, Vector3Stamped
import numpy as np
import numpy.linalg as la
import scipy.spatial.transform

def extractPosition(packet : f1dmsgs.PacketMotionData , car_index = None):
   if car_index is None:
      idx = packet.m_header.player_car_index
   else:
      idx = car_index
   motion_data : f1dmsgs.CarMotionData = packet.car_motion_data[idx]
   position = np.array( (motion_data.world_position.point.x, motion_data.world_position.point.y, motion_data.world_position.point.z), dtype=np.float64)
   return position 

def extractPose(packet : f1dmsgs.PacketMotionData, car_index = None):
   if car_index is None:
      idx = packet.header.player_car_index
   else:
      idx = car_index
   position = extractPosition(packet, car_index=idx)
   motion_data : f1dmsgs.CarMotionData = packet.car_motion_data[idx]

   rightdir : geo_msgs.Vector3 = motion_data.world_right_dir.vector
   forwarddir : geo_msgs.Vector3 = motion_data.world_forward_dir.vector

   rightvector = np.array((rightdir.x, rightdir.y, rightdir.z), dtype=np.float64)
   rightvector = rightvector/la.norm(rightvector)

   forwardvector = np.array((forwarddir.x, forwarddir.y, forwarddir.z), dtype=np.float64)
   forwardvector = forwardvector/la.norm(forwardvector)

   upvector = np.cross(rightvector,forwardvector)
   upvector = upvector/la.norm(upvector)
   rotationmat = np.column_stack((-rightvector,upvector,forwardvector))
   return ( position, scipy.spatial.transform.Rotation.from_matrix(rotationmat).as_quat() )