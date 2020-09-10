"""Launch the cpp_code executable in this package"""

from launch import LaunchDescription
import launch_ros.actions


def generate_launch_description():
    return LaunchDescription([
        launch_ros.actions.Node(package='f1_datalogger_ros', executable='boundary_publisher', output='screen'),
        launch_ros.actions.Node(package='f1_datalogger_ros', executable='tf_updater', output='screen')
    ])