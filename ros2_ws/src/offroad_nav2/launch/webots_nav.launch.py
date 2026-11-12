"""Whole Webots demo: sim + stereo depth + localization + perception + Nav2 (forward-only).
ros2 launch offroad_nav2 webots_nav.launch.py [gui:=true] [tl_source:=none|standin] [barrels_seed:=7]
Perception (three networks, stereo depth) and Nav2 (forward-only, RPP) are always started; obstacles come from the camera only."""
import os
from ament_index_python.packages import get_package_share_directory as share
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression


def generate_launch_description():
    L = LaunchConfiguration
    inc = lambda f, cond=None, **a: IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share('offroad_nav2'), 'launch', f)),
        launch_arguments=a.items(), condition=cond)
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='false'),
        DeclareLaunchArgument('tl_source', default_value='none'),
        DeclareLaunchArgument('barrels_seed', default_value='7', description='>0: fixed barrel layout; 0: leave world random'),
        # driver_odom_tf=0: ekf_local owns odom->base_link
        inc('webots_sensors.launch.py', gui=L('gui'), driver_odom_tf='0',
            standin_tl=PythonExpression(["'1' if '", L('tl_source'), "' == 'standin' else '0'"])),
        inc('localization.launch.py', sim='true'),
        inc('perception.launch.py', tl_source=L('tl_source')),
        inc('navigation.launch.py'),
        TimerAction(period=20.0, actions=[ExecuteProcess(
            cmd=['ros2', 'topic', 'pub', '--once', '/vesta/barrels_seed', 'std_msgs/msg/Int32', ['{data: ', L('barrels_seed'), '}']],
            condition=IfCondition(PythonExpression(["'", L('barrels_seed'), "' != '0'"])), output='log')]),
    ])
