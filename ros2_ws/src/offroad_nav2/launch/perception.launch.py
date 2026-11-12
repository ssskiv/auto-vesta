"""perception_node (C++, three networks via ONNX Runtime + depth -> clouds), the traffic-light node and the stop wall.
All parameters are in config/nodes.yaml; the launch argument tl_source overrides traffic_light_node.source and stop_wall.enabled."""
import os

from ament_index_python.packages import get_package_share_directory as share
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(share('offroad_nav2'), 'config', 'nodes.yaml')
    tl = LaunchConfiguration('tl_source')
    return LaunchDescription([
        DeclareLaunchArgument('tl_source', default_value='none', description='none | standin (world truth, not perception)'),
        Node(package='offroad_nav2', executable='perception_node', name='perception_node', output='screen', parameters=[cfg]),
        Node(package='offroad_nav2', executable='traffic_light_node', name='traffic_light_node', output='screen',
             parameters=[cfg, {'source': tl}]),
        Node(package='offroad_nav2', executable='stop_wall_node', name='stop_wall', output='screen',
             parameters=[cfg, {'enabled': PythonExpression(["'", tl, "' != 'none'"])}]),
    ])
