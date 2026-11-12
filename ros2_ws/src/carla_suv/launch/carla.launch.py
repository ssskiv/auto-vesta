#!/usr/bin/env python3
"""CARLA simulation of the Vesta: carla-ros-bridge + adapters that restore the Webots interface."""
import json
import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess, IncludeLaunchDescription,
                            LogInfo, OpaqueFunction, RegisterEventHandler, TimerAction)
from launch.events import Shutdown
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ROLE = 'vesta'
SENSOR_SETS = {'full': 'vesta_sensors.json', 'stereo': 'vesta_stereo_sensors.json'}


def shifted_objects_file(rear_axle_offset, config):
    """Sensor positions in vesta_sensors.json are relative to base_link (the rear axle, as in
    vesta.urdf). CARLA mounts sensors on the actor origin, so move them by the axle offset."""
    path = os.path.join(get_package_share_directory('carla_suv'), 'config', config)
    with open(path) as f:
        objects = json.load(f)
    for obj in objects['objects']:
        for sensor in obj.get('sensors', []):
            if 'spawn_point' in sensor:   # pseudo sensors have none
                sensor['spawn_point']['x'] -= rear_axle_offset
    out = os.path.join(tempfile.mkdtemp(prefix='vesta_carla_'), 'objects.json')
    with open(out, 'w') as f:
        json.dump(objects, f)
    return out


def launch_setup(context):
    cfg = {k: LaunchConfiguration(k).perform(context) for k in (
        'host', 'port', 'map', 'xodr', 'seed', 'task', 'rear_axle_offset', 'max_steer_angle', 'sensors', 'sim_step')}
    offset = float(cfg['rear_axle_offset'])
    suv_dir = get_package_share_directory('webots_ros2_suv')
    bridge_dir = get_package_share_directory('carla_ros_bridge')
    spawn_dir = get_package_share_directory('carla_spawn_objects')

    urdf = os.path.join(suv_dir, 'resource', 'vesta.urdf')
    with open(urdf) as f:
        robot_description = f.read()

    world_setup = ExecuteProcess(
        cmd=['ros2', 'run', 'carla_suv', 'world_setup', '--host', cfg['host'], '--port', cfg['port'],
             '--map', cfg['map'], '--map-yaml', os.path.join(suv_dir, 'maps', 'robocross.yaml'),
             '--seed', cfg['seed'], '--task', cfg['task']] + (['--xodr', cfg['xodr']] if cfg['xodr'] else []),
        output='screen')

    ticker = ExecuteProcess(
        cmd=['ros2', 'run', 'carla_suv', 'world_setup', '--host', cfg['host'], '--port', cfg['port'], '--ticker', '--step', cfg['sim_step'], '--hold', '12'],
        output='screen')

    bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bridge_dir, 'carla_ros_bridge.launch.py')),
        launch_arguments={
            # Passive: the bridge must not load a town itself (it would replace the map world_setup
            # loaded) and it does not tick; the ticker below does, in real time.
            'host': cfg['host'], 'port': cfg['port'], 'timeout': '240',   # the first spawn on a loaded server is slow
            'synchronous_mode': 'False', 'passive': 'True',
            'ego_vehicle_role_name': ROLE, 'fixed_delta_seconds': cfg['sim_step'],
        }.items())

    spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(spawn_dir, 'carla_spawn_objects.launch.py')),
        launch_arguments={
            'objects_definition_file': shifted_objects_file(offset, SENSOR_SETS[cfg['sensors']]),
            'spawn_sensors_only': 'False',
        }.items())

    adapters = [
        Node(package='carla_suv', executable='cmd_vel_adapter', output='screen',
             parameters=[{'role_name': ROLE, 'carla_max_steer_angle': float(cfg['max_steer_angle'])}]),
        Node(package='carla_suv', executable='sensor_adapter', output='screen',
             parameters=[{'role_name': ROLE, 'rear_axle_offset': offset,
                          'carla_max_steer_angle': float(cfg['max_steer_angle'])}]),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description}], output='screen'),
    ]

    if cfg['sensors'] == 'stereo':
        # Frames of the stereo pair (not in vesta.urdf, which Webots shares), from the same JSON.
        with open(os.path.join(get_package_share_directory('carla_suv'), 'config', SENSOR_SETS['stereo'])) as f:
            for sensor in json.load(f)['objects'][0]['sensors']:
                if sensor['id'].startswith('stereo_'):
                    p = sensor['spawn_point']
                    adapters.append(Node(
                        package='tf2_ros', executable='static_transform_publisher',
                        arguments=['--x', str(p['x']), '--y', str(p['y']), '--z', str(p['z']),
                                   '--frame-id', 'base_link', '--child-frame-id', sensor['id'] + '_link']))

    # The map must be loaded before the bridge connects.
    def after_world_setup(event, _context):
        if event.returncode != 0:   # no map, no barrels: do not run on whatever world the server has
            return [LogInfo(msg='world_setup failed (map not loaded), stopping'), EmitEvent(event=Shutdown())]
        # The ticker freezes the world (hold 12 s) while the sensors are spawned at +5 s (see run_ticker).
        return [ticker, bridge, TimerAction(period=5.0, actions=[spawn] + adapters)]

    return [
        world_setup,
        RegisterEventHandler(OnProcessExit(target_action=world_setup, on_exit=after_world_setup)),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('host', default_value=os.environ.get('CARLA_HOST', 'carlasim')),
        DeclareLaunchArgument('port', default_value=os.environ.get('CARLA_PORT', '2000')),
        DeclareLaunchArgument('map', default_value='Robocross', description='map on the CARLA server'),
        DeclareLaunchArgument('xodr', default_value='',
                              description='OpenDRIVE file (tools/robocross_assets.py) if the map is not on the server'),
        DeclareLaunchArgument('seed', default_value='0', description='barrel seed, <= 0 random'),
        DeclareLaunchArgument('task', default_value='basic', description='barrels: basic or extended'),
        DeclareLaunchArgument('rear_axle_offset', default_value='1.4',
                              description='m from the CARLA actor origin back to the rear axle'),
        DeclareLaunchArgument('max_steer_angle', default_value='1.22',
                              description='rad of front wheel angle at CARLA steer = 1'),
        DeclareLaunchArgument('sensors', default_value='full', choices=list(SENSOR_SETS),
                              description='full: camera, lidar, radar, gnss, imu; stereo: stereo pair, gnss, imu'),
        DeclareLaunchArgument('sim_step', default_value='0.05',
                              description='s per simulation tick, real time. The IMU rate is 1/sim_step: use 0.01 for stereo-inertial'),
        OpaqueFunction(function=launch_setup),
    ])
