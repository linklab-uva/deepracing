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

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from f1_datalogger_msgs.msg import TimestampedPacketMotionData
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped

class DataListener(Node):

    def __init__(self):
        super().__init__('motion_data_listener')
        self.sub = self.create_subscription(TimestampedPacketMotionData, '/motion_data', self.motion_data_callback, 10)
        self.point_pub = self.create_publisher(PointStamped, '/car_position')
        self.pose_pub = self.create_publisher(PoseStamped, '/car_pose')

    def motion_data_callback(self, msg):
        point = msg.udp_packet.car_motion_data[0].world_position
        self.point_pub.publish(point)
        #self.get_logger().info('I heard: [%s]' % msg)


def main(args=None):
    rclpy.init(args=args)

    node = DataListener()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()