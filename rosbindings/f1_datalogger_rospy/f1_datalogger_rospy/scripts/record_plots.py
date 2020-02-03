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
from f1_datalogger_msgs.msg import TimestampedPacketMotionData, CarMotionData, ImageWithPath
from geometry_msgs.msg import PoseStamped, Pose
from geometry_msgs.msg import PointStamped, Point
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib.figure
import matplotlib.image
import matplotlib.axes
import matplotlib.lines
import matplotlib.pyplot as plt
import cv_bridge
from sensor_msgs.msg import Image
from typing import List
import cv2
import os
import shutil
class PlotRecorder(Node):

    def __init__(self):
        super().__init__('plot_recorder', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.sub = self.create_subscription(ImageWithPath, '/predicted_path', self.plot_callback, 1)
        self.pub = self.create_publisher(Image, '/image_plots', 1)
        self.f : matplotlib.figure.Figure = plt.figure()
        subplot_kw_dict : dict = {"xmargin" : 0.0, "ymargin" : 0.0}
        self.axarr : List[matplotlib.axes.Axes] = self.f.subplots(1,2,subplot_kw=subplot_kw_dict)
        self.imdata = self.axarr[0].imshow(np.zeros((66,200,3),dtype=np.uint8))
        self.axarr[1].set_xlabel("Lateral Axis (meters)")
        self.axarr[1].set_xlim(-20,20)
        self.axarr[1].set_ylabel("Forward Axis (meters)", labelpad = -7 )
        self.axarr[1].set_ylim(0,100)
        
        imdirparam : Parameter = self.get_parameter_or("image_directory",Parameter("image_directory", value="plot_images"))
        self.imdir = imdirparam.get_parameter_value().string_value
        overwriteparam : Parameter = self.get_parameter_or("overwrite",Parameter("overwrite", value=False))
        if overwriteparam.get_parameter_value().bool_value and os.path.isdir(self.imdir):
            shutil.rmtree(self.imdir)
        os.makedirs(self.imdir, exist_ok=False)
        self.plotdata : List[matplotlib.lines.Line2D] = self.axarr[1].plot(np.zeros(60), np.zeros(60), label = "Predicted trajectory")
        
        
        self.cvbridge : cv_bridge.CvBridge = cv_bridge.CvBridge()
        self.counter = 1
        
        
    def plot_callback(self, msg : ImageWithPath):
        img_msg = msg.image
        if img_msg.height<=0 or img_msg.width<=0:
            return
        imnp = self.cvbridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8") 
        self.plotdata[0].set_xdata(-1.0*np.array(msg.path.posx))
        self.plotdata[0].set_ydata(np.array(msg.path.posz))
    
        self.imdata.set_data(imnp)
        self.f.canvas.draw()
        imgplotnp = np.fromstring(self.f.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(self.f.canvas.get_width_height()[::-1] + (3,))
        imgplotnp=imgplotnp[50:,50:245,:]
        imgplotmsg = self.cvbridge.cv2_to_imgmsg(imgplotnp,encoding="rgb8")
        self.pub.publish(imgplotmsg)
        cv2.imwrite(os.path.join(self.imdir,"plot_%d.jpg" % (self.counter,) ), cv2.cvtColor(imgplotnp, cv2.COLOR_RGB2BGR))
        self.counter = self.counter + 1
        # show original image
        #self.fig.add_subplot(221)
        #plt.show()

        
        #self.get_logger().info('I heard: [%s]' % msg)


def main(args=None):
    rclpy.init(args=args)

    node = PlotRecorder()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()