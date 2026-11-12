"""Webots + sim-time restamp + stereo depth.  Publishes /stereo/{left,right}/image_rect, camera_info, /stereo/depth."""
import os
from ament_index_python.packages import get_package_share_directory as share
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    cfg = os.path.join(share('offroad_nav2'), 'config', 'stereo.yaml')
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share('webots_ros2_suv'), 'launch', 'robot.launch.py')),
        launch_arguments={'gui': LaunchConfiguration('gui'), 'mode': LaunchConfiguration('mode')}.items(),
        condition=IfCondition(LaunchConfiguration('start_sim')))
    relay = Node(package='offroad_nav2', executable='stamp_relay', name='stamp_relay',
                 parameters=[{'use_sim_time': True}], output='screen')
    disparity = ComposableNodeContainer(
        name='stereo_container', namespace='', package='rclcpp_components', executable='component_container',
        composable_node_descriptions=[ComposableNode(
            package='stereo_image_proc', plugin='stereo_image_proc::DisparityNode', name='stereo_disparity',
            parameters=[cfg, {'use_sim_time': True}],
            remappings=[('left/image_rect', '/stereo_crop/left/image_rect'), ('left/camera_info', '/stereo_crop/left/camera_info'),
                        ('right/image_rect', '/stereo_crop/right/image_rect'),
                        ('right/camera_info', '/stereo_crop/right/camera_info'), ('disparity', '/stereo/disparity')])],
        output='screen')
    depth = Node(package='offroad_nav2', executable='disparity_to_depth', name='disparity_to_depth',
                 parameters=[{'use_sim_time': True}], output='screen')
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='false'),
        DeclareLaunchArgument('mode', default_value='realtime'),
        DeclareLaunchArgument('start_sim', default_value='true'),
        DeclareLaunchArgument('driver_odom_tf', default_value='1'),
        DeclareLaunchArgument('standin_tl', default_value='0'),   # 1: driver publishes the world light state (STAND-IN)
        SetEnvironmentVariable('VESTA_STANDIN_TL', LaunchConfiguration('standin_tl')),
        # the localization stack owns odom->base_link (see webots_nav.launch.py)
        SetEnvironmentVariable('VESTA_PUBLISH_ODOM_TF', LaunchConfiguration('driver_odom_tf')),
        sim, relay, disparity, depth,
    ])
