#!/usr/bin/env python3
import os
import pathlib
import launch
from launch.substitutions import  LaunchConfiguration
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions.path_join_substitution import PathJoinSubstitution
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController
from launch.actions import DeclareLaunchArgument, GroupAction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import LoadComposableNodes, SetParameter
from launch_ros.descriptions import ComposableNode, ParameterFile
#from launch.conditions import IfCondition#, UnlessCondition


from webots_ros2_suv import referee as referee_script
from webots_ros2_suv.runtime_assets import link_runtime_assets

PACKAGE_NAME = 'webots_ros2_suv'
USE_SIM_TIME = True


def generate_launch_description():
    package_dir = get_package_share_directory(PACKAGE_NAME)
    link_runtime_assets(pathlib.Path(package_dir))
    world = LaunchConfiguration('world')

    webots = WebotsLauncher(
        world=PathJoinSubstitution([package_dir, 'worlds', world]),
        mode=LaunchConfiguration('mode'),
        gui=LaunchConfiguration('gui'),
        ros2_supervisor=True,
        stream=True,
    )

    def vehicle_nodes(context):
        buggy = pathlib.Path(world.perform(context)).name == 'robocross_buggy.wbt'
        vehicle = 'buggy' if buggy else 'vesta'
        description_path = os.path.join(package_dir, 'resource', vehicle + '.urdf')
        robot_desc = pathlib.Path(description_path).read_text()
        return [
            WebotsController(
                robot_name=vehicle,
                parameters=[{'robot_description': description_path}],
                respawn=True,
            ),
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                name='robot_state_publisher',
                output='screen',
                parameters=[{'use_sim_time': USE_SIM_TIME, 'robot_description': robot_desc}],
            ),
        ]

    # Automatic referee (РобоКросс.Трасса-2026 scoring), extern controller of the Robot "referee"
    referee = launch.actions.ExecuteProcess(
        cmd=[os.path.join(os.environ.get('WEBOTS_HOME', '/usr/local/webots'), 'webots-controller'),
             '--robot-name=referee', referee_script.__file__],
        output='screen', respawn=True,
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value='robocross.wbt',
            description='Robocross simulation world'
        ),
        DeclareLaunchArgument(
            'mode',
            default_value='realtime',
            description='Webots run mode: realtime or fast'
        ),
        DeclareLaunchArgument(
            'gui',
            default_value='true',
            description='false runs Webots without rendering'
        ),
        # WebotsLauncher rewrites the world launch configuration on execution.
        OpaqueFunction(function=vehicle_nodes),
        webots,
        webots._supervisor,
        referee,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        )
    ])
