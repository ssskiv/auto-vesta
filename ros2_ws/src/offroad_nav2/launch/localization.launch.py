"""IMU + GNSS + wheel speed -> odom->base_link (ekf_local), navsat_transform, map->odom (ekf_global).
sim:=true adds the Webots sensor adapter (PointStamped GPS -> NavSatFix, IMU covariances).
Parameters: config/nodes.yaml (sim_sensor_adapter, wheel_odometry), config/ekf.yaml, config/navsat.yaml."""
import os
from ament_index_python.packages import get_package_share_directory as share
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cfg = lambda f: os.path.join(share('offroad_nav2'), 'config', f)
    return LaunchDescription([
        DeclareLaunchArgument('sim', default_value='true'),
        Node(package='offroad_nav2', executable='sim_sensor_adapter', parameters=[cfg('nodes.yaml')],
             condition=IfCondition(LaunchConfiguration('sim')), output='screen'),
        Node(package='offroad_nav2', executable='wheel_odometry', parameters=[cfg('nodes.yaml')], output='screen'),
        Node(package='robot_localization', executable='ekf_node', name='ekf_local', parameters=[cfg('ekf.yaml')],
             remappings=[('odometry/filtered', '/odometry/local')], output='screen'),
        Node(package='robot_localization', executable='navsat_transform_node', name='navsat_transform',
             parameters=[cfg('navsat.yaml')],
             remappings=[('imu/data', '/imu/data_cov'), ('gps/fix', '/gps/fix_ll'),
                         ('odometry/filtered', '/odometry/global'), ('odometry/gps', '/odometry/gps')], output='screen'),
        Node(package='robot_localization', executable='ekf_node', name='ekf_global', parameters=[cfg('ekf.yaml')],
             remappings=[('odometry/filtered', '/odometry/global')], output='screen'),
    ])
