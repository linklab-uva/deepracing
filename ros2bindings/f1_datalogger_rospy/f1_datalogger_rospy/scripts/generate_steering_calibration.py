import rclpy
from rclpy.node import Node
from f1_datalogger_msgs.msg import PacketHeader
from f1_datalogger_msgs.msg import TimestampedPacketMotionData, PacketMotionData, CarMotionData
import py_f1_interface
import numpy as np
import scipy, scipy.stats
class SteeringCalibrationNode(Node):
    def __init__(self):
        super(SteeringCalibrationNode, self).__init__('generate_steering_calibration', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.controller = py_f1_interface.F1Interface(1)
        self.controller.setControl(0.0,0.0,0.0)
        self.current_motion_data = None
    def motionDataCB(self, msg):
        self.current_motion_data = msg

def main(args=None):
    rclpy.init(args=args)
    rclpy.logging.initialize()
    node = SteeringCalibrationNode()
    # executor = rclpy.executors.MultiThreadedExecutor(num_threads=5)
    # executor.add_node(node)
    rate = node.create_rate(1/5.0)
    node.create_subscription(TimestampedPacketMotionData, "/motion_data", node.motionDataCB, 1)

    negative_vjoy_angles = np.linspace(-1.0,0.0,num=50,dtype=np.float64)
    negative_wheel_angles = np.zeros_like(negative_vjoy_angles)

    positive_vjoy_angles = np.linspace(0.0,1.0,num=50,dtype=np.float64)
    positive_wheel_angles = np.zeros_like(positive_vjoy_angles)
    print("Sleeping until initial wheel data arrives")
    #rclpy.spin_once(node)
    #rate.sleep()
    while node.current_motion_data is None:
        rclpy.spin_once(node)
    [rclpy.spin_once(node) for j in range(30)]
    try:
        for i in range(negative_vjoy_angles.shape[0]):
            angle = negative_vjoy_angles[i]
            node.controller.setControl(angle,0.0,0.0)
            print("set angle to %f" %(angle,))
            [rclpy.spin_once(node) for j in range(30)]
            negative_wheel_angles[i] = -node.current_motion_data.udp_packet.front_wheels_angle
         #   rate.sleep()
           # executor.spin_once()
        for i in range(positive_vjoy_angles.shape[0]):
            angle = positive_vjoy_angles[i]
            node.controller.setControl(angle,0.0,0.0)
            print("set angle to %f" %(angle,))
            [rclpy.spin_once(node) for j in range(30)]
            positive_wheel_angles[i] = -node.current_motion_data.udp_packet.front_wheels_angle
           # rate.sleep()
           # executor.spin_once()
    except KeyboardInterrupt:
        rclpy.shutdown()
    slope_neg, intercept_neg, r_value_neg, p_value_neg, std_err_neg = scipy.stats.linregress(negative_wheel_angles, negative_vjoy_angles)
    print("Negative range slope: %f. Negative range intercept: %f" % (slope_neg, intercept_neg))
    slope_pos, intercept_pos, r_value_pos, p_value_pos, std_err_pos = scipy.stats.linregress(positive_wheel_angles, positive_vjoy_angles)
    print("Positive range slope: %f. Positive range intercept: %f" % (slope_pos, intercept_pos))
    


if __name__ == '__main__':
    main()

    
# parser.add_argument("inner_track", type=str,  help="Json file for the inner boundaries of the track.")
# parser.add_argument("outer_track", type=str,  help="Json file for the outer boundaries of the track.")