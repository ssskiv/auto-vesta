"""Nav2 (forward-only, Regulated Pure Pursuit) + velocity smoother + collision monitor + zero-clamp relay.
Obstacles come from the perception clouds."""
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory as share
from launch import LaunchDescription
from launch.actions import OpaqueFunction, TimerAction
from launch_ros.actions import Node

from offroad_bringup.nav2_params import build


def setup(context):
    bt = os.path.join(share('offroad_nav2'), 'behavior_trees')
    map_yaml = os.path.join(share('webots_ros2_suv'), 'maps', 'robocross.yaml')
    params = build(bt, map_yaml)
    f = tempfile.NamedTemporaryFile('w', suffix='_nav2.yaml', delete=False)
    yaml.safe_dump(params, f)
    f.close()
    pf = [f.name]
    sim = {'use_sim_time': True}
    return [
        Node(package='nav2_map_server', executable='map_server', name='map_server', parameters=pf, output='screen'),
        Node(package='nav2_planner', executable='planner_server', name='planner_server', parameters=pf, output='screen'),
        Node(package='nav2_controller', executable='controller_server', name='controller_server', parameters=pf,
             remappings=[('cmd_vel', 'cmd_vel_nav_raw'), ('odom', '/odometry/local')], output='screen'),
        Node(package='nav2_behaviors', executable='behavior_server', name='behavior_server', parameters=pf, output='screen'),
        Node(package='nav2_bt_navigator', executable='bt_navigator', name='bt_navigator', parameters=pf, output='screen'),
        Node(package='nav2_waypoint_follower', executable='waypoint_follower', name='waypoint_follower', parameters=pf,
             output='screen'),
        Node(package='nav2_velocity_smoother', executable='velocity_smoother', name='velocity_smoother', parameters=pf,
             remappings=[('cmd_vel', 'cmd_vel_nav_raw'), ('cmd_vel_smoothed', 'cmd_vel_smoothed')], output='screen'),
        Node(package='nav2_collision_monitor', executable='collision_monitor', name='collision_monitor', parameters=pf,
             output='screen'),
        Node(package='offroad_nav2', executable='zero_clamp_relay', name='zero_clamp_relay',
             parameters=[{'use_sim_time': True, 'in_topic': '/cmd_vel_nav', 'out_topic': '/cmd_vel'}], output='screen'),
        # started a few seconds later: if it races the nodes' DDS discovery, a lost change_state response hangs it
        TimerAction(period=20.0, actions=[Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                                              name='lifecycle_manager_navigation', parameters=pf, output='screen')]),
    ]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=setup)])
