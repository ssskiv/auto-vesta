#!/usr/bin/env python3
import os
import pathlib
import launch
from launch.substitutions import  LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.substitutions.path_join_substitution import PathJoinSubstitution
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController
from nav2_common.launch import RewrittenYaml
from launch.actions import DeclareLaunchArgument, GroupAction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import LoadComposableNodes, SetParameter
from launch_ros.descriptions import ComposableNode, ParameterFile
#from launch.conditions import IfCondition#, UnlessCondition


PACKAGE_NAME = 'webots_ros2_suv'
USE_SIM_TIME = True


def generate_launch_description():
    package_dir = get_package_share_directory(PACKAGE_NAME)
    world = LaunchConfiguration('world')

    webots = WebotsLauncher(world=PathJoinSubstitution([package_dir, 'worlds', world]), ros2_supervisor=True, stream=True)
    robot_description_path = os.path.join(package_dir, pathlib.Path(os.path.join(package_dir, 'resource', 'vesta.urdf')))
    vehicle_driver = WebotsController(
        robot_name='vesta',
        parameters=[
            {'robot_description': robot_description_path}
        ],
        respawn=True
    )

    with open(robot_description_path, 'r') as infp:
        robot_desc = infp.read()
    state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': USE_SIM_TIME, 'robot_description': robot_desc}],
    )
    

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value='robocross.wbt',
            description='Robocross simulation world'
        ),
        webots,
        webots._supervisor,
        vehicle_driver,
        state_publisher_node,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        )
    ])