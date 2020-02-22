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
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from f1_datalogger_msgs.msg import TimestampedPacketMotionData, CarMotionData
from geometry_msgs.msg import PoseStamped, Pose
from geometry_msgs.msg import PointStamped, Point
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from f1_datalogger_rospy.controls.pure_puresuit_control_ros import PurePursuitControllerROS
from f1_datalogger_rospy.controls.pure_puresuit_control_waypoint_predictor import AdmiralNetWaypointPredictorROS

def main(args=None):
    rclpy.init(args=args)
    rclpy.logging.initialize()
    node = AdmiralNetWaypointPredictorROS()
    node.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
    node.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    node.destroy_node()
    rclpy.shutdown()
    


if __name__ == '__main__':
    main()