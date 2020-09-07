import sys
import os
import rclpy 
from rclpy.node import Node
import sensor_msgs.msg as sensor_msgs
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import sys
from collections import namedtuple
import ctypes
import math
import struct
import f1_datalogger_rospy.convert


class PCDListener(Node):

    def __init__(self):
        super().__init__('pcd_subsriber_node')

        self.pcd_subscriber = self.create_subscription(
            sensor_msgs.PointCloud2,    # Msg type
            '/optimal_raceline/pcl',                      # topic
            self.listener_callback,      # Function to call
            10                          # QoS
        )

                
    def listener_callback(self, msg):
       # field_names=["x","y","z"]
        field_names=None

        print("Converting a message to numpy")
        pcd_as_numpy_array = np.array(list(f1_datalogger_rospy.convert.pointCloud2ToNumpy(msg, field_names=field_names)))
        print(pcd_as_numpy_array)
        
 
# https://github.com/ros2/common_interfaces


def main(args=None):
    # Boilerplate code.
    rclpy.init(args=args)
    pcd_listener = PCDListener()
    rclpy.spin(pcd_listener)
    pcd_listener.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()