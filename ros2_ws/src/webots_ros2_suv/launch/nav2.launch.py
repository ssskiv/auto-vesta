#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml


PACKAGE_NAME = 'webots_ros2_suv'
USE_SIM_TIME = True


def generate_launch_description():
    package_dir = get_package_share_directory(PACKAGE_NAME)

    map_yaml = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')

    lifecycle_nodes = [
        'map_server',
        'planner_server',
        'controller_server',
    ]

    configured_params = ParameterFile(
        RewrittenYaml(
            source_file=params_file,
            param_rewrites={
                'use_sim_time': use_sim_time,
                'yaml_filename': map_yaml,
            },
            convert_types=True),
        allow_substs=True)

    # map -> odom: odom is anchored to the first GPS fix and does not drift
    # (see VestaDriver.__base_link_transform), so if the map was recorded from
    # the same spawn pose the two frames coincide. Use the offset args if the
    # map origin and the spawn point ever differ.
    map_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_odom_tf',
        output='screen',
        arguments=[
            '--x', LaunchConfiguration('map_odom_x'),
            '--y', LaunchConfiguration('map_odom_y'),
            '--yaw', LaunchConfiguration('map_odom_yaw'),
            '--frame-id', 'map',
            '--child-frame-id', 'odom',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[configured_params],
    )

    # planner_server owns global_costmap, controller_server owns local_costmap
    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[configured_params],
    )

    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[configured_params],
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': lifecycle_nodes,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=os.path.join(package_dir, 'maps', 'robocross.yaml'),
            description='Full path to the map yaml file'
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(package_dir, 'config', 'nav2_params.yaml'),
            description='Full path to the Nav2 parameters file'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value=str(USE_SIM_TIME).lower(),
            description='Use simulation (Webots) clock'
        ),
        DeclareLaunchArgument('map_odom_x', default_value='0.0'),
        DeclareLaunchArgument('map_odom_y', default_value='0.0'),
        DeclareLaunchArgument('map_odom_yaw', default_value='0.0'),
        map_odom_tf,
        map_server,
        planner_server,
        controller_server,
        lifecycle_manager,
    ])
