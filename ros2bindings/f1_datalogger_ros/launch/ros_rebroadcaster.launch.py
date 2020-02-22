"""Launch the cpp_code executable in this package"""

from launch import LaunchDescription
import launch_ros.actions


def generate_launch_description():
    rebroadcasternode = launch_ros.actions.Node(package='f1_datalogger_ros', node_executable='ros_rebroadcaster', output='screen', node_name="f1_data_broadcaster")
    return LaunchDescription([rebroadcasternode,]) 