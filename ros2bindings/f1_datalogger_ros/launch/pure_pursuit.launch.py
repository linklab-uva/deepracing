"""Launch the cpp_code executable in this package"""

from launch import LaunchDescription
import launch_ros.actions


def generate_launch_description():
    tfupdater = launch_ros.actions.Node(package='f1_datalogger_ros', node_executable='tf_updater', node_name="tf_updater",output="screen")
    pathconverter = launch_ros.actions.Node(package='f1_datalogger_ros', node_executable='path_converter', node_name="path_converter")
    posepublishernode = launch_ros.actions.Node(package='f1_datalogger_rospy', node_executable='pose_publisher', node_name="f1_data_broadcaster",output="screen")
    purepursuitnode = launch_ros.actions.Node( package='f1_datalogger_rospy', node_executable='pure_pursuit_control', node_name="pure_pursuit_control",  parameters=["D:/deepracing/DCNN-Pytorch/uva/rosparams/purepursuit_ros.yaml",],output="screen")
    rviz = launch_ros.actions.Node( package='rviz2', node_executable='rviz2', node_name="rviz" )

    nodelist = [purepursuitnode]
    nodelist+=[tfupdater]
    #nodelist+=[posepublishernode]
    #nodelist+=[pathconverter]
    #nodelist+=[rviz]
    return LaunchDescription(nodelist) 