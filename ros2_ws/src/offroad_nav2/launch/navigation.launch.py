"""Nav2 (forward-only, Regulated Pure Pursuit) + velocity smoother + collision monitor + zero-clamp relay.
Parameters: config/nav2_params.yaml (Nav2) and config/nodes.yaml (zero_clamp_relay). Only the map file and the two behaviour-tree files
are substituted here (paths in the package share directory). Obstacles come from the perception clouds."""
import os

from ament_index_python.packages import get_package_share_directory as share
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    here = share('offroad_nav2')
    bt = os.path.join(here, 'behavior_trees')
    pf = [ParameterFile(RewrittenYaml(
        source_file=os.path.join(here, 'config', 'nav2_params.yaml'), root_key='',
        param_rewrites={'yaml_filename': os.path.join(share('webots_ros2_suv'), 'maps', 'robocross.yaml'),
                        'default_nav_to_pose_bt_xml': os.path.join(bt, 'nav_forward_only.xml'),
                        'default_nav_through_poses_bt_xml': os.path.join(bt, 'nav_through_poses_forward_only.xml')},
        convert_types=True), allow_substs=True)]
    return LaunchDescription([
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
             parameters=[os.path.join(here, 'config', 'nodes.yaml')], output='screen'),
        # started a few seconds later: if it races the nodes' DDS discovery, a lost change_state response hangs it
        TimerAction(period=20.0, actions=[Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                                              name='lifecycle_manager_navigation', parameters=pf, output='screen')]),
    ])
