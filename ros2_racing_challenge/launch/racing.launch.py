import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory('ros2_racing_challenge')
    rviz_config = os.path.join(package_share, 'rviz', 'racing.rviz')

    return LaunchDescription([
        Node(
            package='ros2_racing_challenge',
            executable='racing_sim',
            name='racing_sim',
            output='screen',
        ),
        Node(
            package='ros2_racing_challenge',
            executable='racing_controller',
            name='racing_controller',
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
        ),
    ])
