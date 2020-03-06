# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy, rclpy.clock, rclpy.time
from rclpy.node import Node
from rclpy import Parameter
from std_msgs.msg import String
from f1_datalogger_msgs.msg import TimestampedPacketMotionData, CarMotionData
from geometry_msgs.msg import PoseStamped, Pose
from geometry_msgs.msg import PointStamped, Point, Point32
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import PointCloud2, PointCloud, PointField, PointField
from std_msgs.msg import Header
import deepracing, deepracing.protobuf_utils as proto_utils

class DataListener(Node):

    def __init__(self):
        super().__init__('pose_publisher', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.sub = self.create_subscription(TimestampedPacketMotionData, '/motion_data', self.motion_data_callback, 1)
        self.point_pub = self.create_publisher(PointStamped, '/car_position', 1)
        self.pose_pub = self.create_publisher(PoseStamped, '/car_pose', 1)
        self.pc_pub = self.create_publisher(PointCloud, '/track_boundaries', rclpy.qos.qos_profile_sensor_data)
        self.rosclock : rclpy.clock.ROSClock = rclpy.clock.ROSClock()


        self.trackfilesparam = self.get_parameter_or( "trackfiles", Parameter(name="trackfiles", type_=Parameter.Type.STRING_ARRAY, value=[]) )
        trackfiles = self.trackfilesparam.get_parameter_value().string_array_value
        if len(trackfiles)==0:
            print("Not loading a trackfile, because the trackfiles param was not set")
        else:
            trackfiles = self.trackfilesparam.get_parameter_value().string_array_value

            print("Using trackfiles: %s" % ( str(trackfiles), ) )
            rinner, Xinner = proto_utils.loadTrackfile(trackfiles[0])
            router, Xouter = proto_utils.loadTrackfile(trackfiles[1])
            self.Xinner = Xinner.astype(np.float32)
            self.Xouter = Xouter.astype(np.float32)
            #now = self.rosclock.now()
            now = rclpy.time.Time()
            self.pcinner : PointCloud  = PointCloud (header=Header(frame_id="track"))
            self.pcouter : PointCloud  = PointCloud (header=Header(frame_id="track"))
            self.pcall : PointCloud  = PointCloud (header=Header(frame_id="track"))
            for i in range(Xinner.shape[0]):
                self.pcinner.points.append(Point32(x=float(self.Xinner[i,0]), y=float(self.Xinner[i,1]), z=float(self.Xinner[i,2])))
                self.pcall.points.append(Point32(x=float(self.Xinner[i,0]), y=float(self.Xinner[i,1]), z=float(self.Xinner[i,2])))
            for i in range(Xouter.shape[0]):
                self.pcouter.points.append(Point32(x=float(self.Xouter[i,0]), y=float(self.Xouter[i,1]), z=float(self.Xouter[i,2])))
                self.pcall.points.append(Point32(x=float(self.Xouter[i,0]), y=float(self.Xouter[i,1]), z=float(self.Xouter[i,2])))



    def motion_data_callback(self, msg):
        motion_data : CarMotionData  = msg.udp_packet.car_motion_data[0]
        pointstamped : PointStamped = motion_data.world_position
        up_vec_ros = motion_data.world_up_dir.vector
        right_vec_ros = motion_data.world_right_dir.vector
        forward_vec_ros = motion_data.world_forward_dir.vector

        up_vec = np.array((up_vec_ros.x, up_vec_ros.y, up_vec_ros.z))
        left_vec = -1.0*np.array((right_vec_ros.x, right_vec_ros.y, right_vec_ros.z))
        forward_vec = np.array((forward_vec_ros.x, forward_vec_ros.y, forward_vec_ros.z))
        rotmat = np.vstack((left_vec, up_vec, forward_vec)).transpose()
        rot = Rot.from_matrix(rotmat)
        quatnp = rot.as_quat()

        pose : Pose = Pose()
        pose.position = pointstamped.point
        pose.orientation.x = quatnp[0]
        pose.orientation.y = quatnp[1]
        pose.orientation.z = quatnp[2]
        pose.orientation.w = quatnp[3]

        self.point_pub.publish(pointstamped)

        posestamped = PoseStamped(header=pointstamped.header, pose=pose)
        self.pose_pub.publish(posestamped)
        #self.get_logger().info('I heard: [%s]' % msg)


def main(args=None):
    rclpy.init(args=args)

    node = DataListener()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    node.pc_pub.publish(node.pcall)
    executor.spin_once()

    
    executor.spin()
    

    
    executor.spin()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()