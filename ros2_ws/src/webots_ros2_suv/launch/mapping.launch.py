#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


PACKAGE_NAME = 'webots_ros2_suv'
USE_SIM_TIME = True


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    # /lidar (VLP-16 PointCloud2 in lidar_link) -> /scan (LaserScan).
    # Heights are in base_link, which sits 0.4 m above ground, so the band below
    # is 0.4..2.0 m above ground (the costmaps in config/nav2_params.yaml
    # start lower, at 0.2 m).
    pointcloud_to_laserscan = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        output='screen',
        remappings=[
            ('cloud_in', '/lidar'),
            ('scan', '/scan'),
        ],
        parameters=[{
            'use_sim_time': use_sim_time,
            'target_frame': 'base_link',
            'transform_tolerance': 0.1,
            'min_height': 0.0,
            'max_height': 1.6,
            'angle_min': -3.14159,
            'angle_max': 3.14159,
            'angle_increment': 0.0035,  # 2*pi / 1800 (lidar horizontalResolution)
            'scan_time': 0.1,
            'range_min': 0.5,
            'range_max': 100.0,  # lidar maxRange
            'use_inf': True,
            'inf_epsilon': 1.0,
        }],
    )

    # Publishes map -> odom itself, so don't run it together with nav2.launch.py.
    slam_toolbox = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        condition=IfCondition(LaunchConfiguration('slam')),
        parameters=[
            os.path.join(get_package_share_directory('slam_toolbox'),
                         'config', 'mapper_params_online_async.yaml'),
            {
                'use_sim_time': use_sim_time,
                'odom_frame': 'odom',
                'map_frame': 'map',
                'base_frame': 'base_link',
                'scan_topic': '/scan',
                'mode': 'mapping',
                'resolution': 0.2,
                'max_laser_range': 40.0,
                'minimum_travel_distance': 0.5,
                'minimum_travel_heading': 0.3,
                'map_update_interval': 2.0,
            },
        ],
    )

    # Saves the current /map on request (see maps/README.md):
    #   ros2 service call /map_saver/save_map nav2_msgs/srv/SaveMap \
    #     "{map_topic: map, map_url: <path>/robocross, image_format: pgm, map_mode: trinary,
    #       free_thresh: 0.25, occupied_thresh: 0.65}"
    map_saver = Node(
        package='nav2_map_server',
        executable='map_saver_server',
        name='map_saver',
        output='screen',
        condition=IfCondition(LaunchConfiguration('slam')),
        parameters=[{
            'use_sim_time': use_sim_time,
            'save_map_timeout': 5.0,
            'free_thresh_default': 0.25,
            'occupied_thresh_default': 0.65,
            'map_subscribe_transient_local': True,
        }],
    )

    # map_saver_server is a lifecycle node and does nothing until activated.
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_mapping',
        output='screen',
        condition=IfCondition(LaunchConfiguration('slam')),
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': ['map_saver'],
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value=str(USE_SIM_TIME).lower(),
            description='Use simulation (Webots) clock'
        ),
        DeclareLaunchArgument(
            'slam',
            default_value='true',
            description='Also run slam_toolbox (online async) on /scan'
        ),
        pointcloud_to_laserscan,
        slam_toolbox,
        map_saver,
        lifecycle_manager,
    ])
