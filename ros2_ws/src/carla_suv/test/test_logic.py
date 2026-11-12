"""Pure-logic tests; they need no CARLA server and no ROS graph."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP_YAML = os.path.join(os.path.dirname(__file__), '..', '..', 'webots_ros2_suv', 'maps', 'robocross.yaml')


def test_barrels_deterministic_and_on_track():
    pytest.importorskip('cv2')
    from carla_suv import world_setup as ws
    free, res, origin = ws.load_map(MAP_YAML)
    a = ws.barrel_positions(free, res, origin, 7, 'basic')
    assert a == ws.barrel_positions(free, res, origin, 7, 'basic')
    assert ws.COUNTS['basic'][0] <= len(a) <= ws.COUNTS['basic'][1]
    assert all(math.hypot(x - ws.SPAWN[0], y - ws.SPAWN[1]) >= ws.SPAWN_SAFE for x, y in a)


def test_cmd_vel_steering_matches_webots_driver():
    pytest.importorskip('carla_msgs')
    from carla_suv.cmd_vel_adapter import MAX_STEER, steer_angle
    assert steer_angle(2.0, 0.0) == 0.0
    assert steer_angle(2.0, 0.3) > 0.0                 # left twist -> left-positive angle
    assert steer_angle(0.5, 5.0) == MAX_STEER          # clamped


def test_stereo_config_is_a_level_pair_with_imu_and_gnss():
    import json
    import os
    path = os.path.join(os.path.dirname(__file__), '..', 'config', 'vesta_stereo_sensors.json')
    sensors = {s['id']: s for s in json.load(open(path))['objects'][0]['sensors']}
    left, right = sensors['stereo_left']['spawn_point'], sensors['stereo_right']['spawn_point']
    assert abs((left['y'] - right['y']) - 0.20) < 1e-9          # baseline
    assert all(left[k] == right[k] for k in ('x', 'z', 'roll', 'pitch', 'yaw'))
    assert sensors['stereo_left']['fov'] == sensors['stereo_right']['fov']
    assert {'gnss', 'imu', 'odometry'} <= set(sensors)
    assert not any('lidar' in i or 'radar' in i for i in sensors)
