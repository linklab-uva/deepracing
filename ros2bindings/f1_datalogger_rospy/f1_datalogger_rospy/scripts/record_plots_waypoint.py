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
from f1_datalogger_msgs.msg import TimestampedPacketMotionData, CarMotionData, ImageWithPath, BezierCurve as BCMessage, PathRaw
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
from typing import List
import cv2
import os
import shutil
import numpy as np
import torch
from deepracing_models.math_utils import bezier
import rclpy.parameter
from rclpy.parameter import Parameter
class PlotRecorder(Node):

    def __init__(self):
        super().__init__('plot_recorder', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.sub = self.create_subscription(PathRaw, '/predicted_path', self.plot_callback, 1)
        self.imsub = self.create_subscription(CompressedImage, '/f1_screencaps/compressed', self.im_callback, 1)
        self.pub = self.create_publisher(Image, '/image_plots', 1)
        self.f : matplotlib.figure.Figure = plt.figure()
        self.canvas : FigureCanvas = FigureCanvas(self.f)
        subplot_kw_dict : dict = {"xmargin" : 0.0, "ymargin" : 0.0}
        self.ax : matplotlib.axes.Axes = self.f.add_subplot()

        plot_images_param = self.get_parameter_or("plot_images", Parameter("plot_images", value=True) )
        self.plot_images = plot_images_param.get_parameter_value().bool_value

        resize_factor_param = self.get_parameter_or("resize_factor", Parameter("resize_factor",value=0.5) )
        self.resize_factor = resize_factor_param.get_parameter_value().double_value

        x_scale_param = self.get_parameter_or("x_scale", Parameter("x_scale",value=10.0) )
        self.x_scale = x_scale_param.get_parameter_value().double_value

        z_scale_param = self.get_parameter_or("z_scale", Parameter("z_scale",value=1.5) )
        self.z_scale = z_scale_param.get_parameter_value().double_value

        self.current_waypoints = None
        self.s = torch.linspace(0,1,60).unsqueeze(0).float()
        
                
        self.cvbridge : cv_bridge.CvBridge = cv_bridge.CvBridge()
        self.counter = 1
        
        image_folder_param = self.get_parameter_or("image_folder", Parameter("image_folder",value="plot_images_waypoint") )
        self.image_folder = image_folder_param.get_parameter_value().string_value
        
        os.makedirs( self.image_folder, exist_ok=True )
        
    def im_callback(self, msg : CompressedImage):
        self.get_logger().debug('Got an image')
        img = cv2.resize(self.cvbridge.compressed_imgmsg_to_cv2(msg, desired_encoding="rgb8")[32:],None, fx=self.resize_factor, fy=self.resize_factor)
        self.get_logger().debug('Decompressed the image')
        if self.current_waypoints is None:
            self.get_logger().info('New waypoints not yet received, skipping')
            return
        waypoints = self.current_waypoints.copy()
        rows = img.shape[0]
        cols = img.shape[1]
        x = range(min(rows,cols))
        self.ax.clear()
        if self.plot_images:
            self.ax.imshow(img, extent=[int(-cols/2), int(cols/2), 0, rows])
            self.ax.scatter(-self.x_scale*waypoints[:,0], self.z_scale*waypoints[:,1], color='blue', label='Predicted Waypoints')
            legendloc='lower right'
        else:
            self.ax.set_xlim([-25,25])
            #self.ax.set_ylim([0,125])
            self.ax.scatter(waypoints[:,0], waypoints[:,1], color='blue', label='Predicted Waypoints')
            legendloc='upper right'
        self.ax.legend(loc=legendloc)
        self.canvas.draw()
        imgout = np.fromstring(self.canvas.tostring_rgb(), dtype='uint8').reshape(self.canvas.get_width_height()[::-1] + (3,))
        msgout = self.cvbridge.cv2_to_imgmsg(imgout,encoding="rgb8")
        impath = os.path.join(self.image_folder, "image_%d.jpg" % (self.counter,) )
        cv2.imwrite(impath,cv2.cvtColor(imgout,cv2.COLOR_RGB2BGR))
        self.counter+=1
        self.pub.publish(msgout)
            
    def plot_callback(self, msg : PathRaw):
        #self.get_logger().info('Got a path: %s' % (msg) )
        self.current_waypoints = np.vstack((msg.posx, msg.posz)).transpose()
        
        #self.get_logger().info('I heard: [%s]' % msg)


def main(args=None):
    rclpy.init(args=args)

    node = PlotRecorder()
    mt_ex : rclpy.executors.MultiThreadedExecutor = rclpy.executors.MultiThreadedExecutor(2)
    mt_ex.add_node(node)
    mt_ex.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()