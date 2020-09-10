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
from rclpy import Parameter
from rclpy.node import Node
from rclpy.time import Time
from rclpy.clock import Clock, ROSClock

from std_msgs.msg import String
from sensor_msgs.msg import Image
from f1_datalogger_msgs.msg import TimestampedPacketMotionData, CarMotionData, ImageWithPath, BezierCurve as BCMessage
from geometry_msgs.msg import PoseStamped, Pose
from geometry_msgs.msg import PointStamped, Point
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.figure
import matplotlib.image
import matplotlib.axes
import matplotlib.lines
import matplotlib.pyplot as plt
import cv_bridge
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, Pose, Vector3, Quaternion, PoseStamped
from f1_datalogger_msgs.msg import BezierCurve
from typing import List
import cv2
import os
import shutil
import numpy as np
import torch
from deepracing_models.math_utils import bezier
import rclpy.parameter
from rclpy.parameter import Parameter
import deepracing_models.math_utils as mu
from copy import deepcopy


class PathPublisher(Node):

    def __init__(self):
        super().__init__('bezier_curve_rviz_feeder')

        num_sample_points_param  : Parameter = self.declare_parameter("num_sample_points", value=10)
        self.num_sample_points : int = num_sample_points_param.get_parameter_value().integer_value
       # print("Number of sample points: %d" %(self.num_sample_points,))

        self.s = torch.linspace(0.0, 1.0, steps=self.num_sample_points, dtype=torch.float64).unsqueeze(0)
        self.A = None
       # self.Adot = None

        self.bcsub = self.create_subscription(BezierCurve,"/input", self.bcCallback, 1)
        self.pathpub = self.create_publisher(Path, "/predicted_path_viz", 1)
        
    def bcCallback(self, msg : BezierCurve):
        if(not (len(msg.control_points_forward) == len(msg.control_points_lateral))):
            self.get_logger().error("Invalid Bezier Curve received, forward dimension has %d control point values, but lateral dimension has %d control point values" (len(msg.control_points_forward) ,len(msg.control_points_lateral) ))
            return
            
        bezier_order = len(msg.control_points_forward)-1
        if (self.A is None) or (not (self.A.shape[2]==bezier_order)):
            self.A = mu.bezierM(self.s, bezier_order).double()
            #self.Adot = mu.bezierM(self.s, bezier_order-1).double()
        bc_torch = torch.stack([torch.tensor(msg.control_points_lateral), torch.tensor(msg.control_points_forward)], dim=1).unsqueeze(0).double()
        pointsxz = torch.matmul(self.A, bc_torch)
        pointsxz = pointsxz[0]
        pointsaug = torch.stack([pointsxz[:,0], msg.yoffset*torch.ones_like(pointsxz[:,0]), pointsxz[:,1], torch.ones_like(pointsxz[:,0])], dim=0)
        # _, velsxz = mu.bezierDerivative(bc_torch,M=self.Adot,order=1)
        # velsxz = velsxz[0]
        # velsaug = torch.stack([velsxz[:,0], torch.zeros_like(velsxz[:,0]), velsxz[:,1]], dim=0)
        # unitvelsaug = velsaug/torch.norm(velsaug,p=2,dim=0)[None,:]
        header = deepcopy(msg.header)
        header.stamp = self.get_clock().now().to_msg()
        pathout = Path(header = header, poses = [PoseStamped(header=header, pose=Pose(position=Point(x=pointsaug[0,i].item(), y=pointsaug[1,i].item(), z=pointsaug[2,i].item()), orientation=Quaternion(x=0.0,y=0.0,z=0.0,w=1.0))) for i in range(pointsaug.shape[1])])
        self.pathpub.publish(pathout)

            
        


def main(args=None):
    rclpy.init(args=args)

    node = PathPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()