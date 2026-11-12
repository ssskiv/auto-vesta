#!/usr/bin/env python3
"""Stop-line test with the STAND-IN light state: log (sim t, light state, x, speed) every 0.5 s while a goal beyond the stop line
is active. Stop line: map x = 9.20; the car's front bumper is base_link + 3.68 m."""
import subprocess
import sys
import time

import rclpy
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
from rclpy.node import Node

rclpy.init()
n = Node('tl_test', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', value=True)])
S = {'state': '?', 'x': 0.0, 'v': 0.0, 'wall': -1}
n.create_subscription(PointCloud2, '/perception/stop_wall', lambda m: S.__setitem__('wall', m.width), 5)
n.create_subscription(String, '/standin/traffic_light_state', lambda m: S.__setitem__('state', m.data), 5)
n.create_subscription(Odometry, '/odometry/global', lambda m: (S.__setitem__('x', m.pose.pose.position.x), S.__setitem__('v', m.twist.twist.linear.x)), 5)
while S['state'] != 'GREEN':   # wait for the NEXT red phase: green first, then the transition to red
    rclpy.spin_once(n, timeout_sec=0.1)
while S['state'] != 'RED':
    rclpy.spin_once(n, timeout_sec=0.1)
print('light turned RED, sending goal beyond the stop line', flush=True)
goal = subprocess.Popen(['ros2', 'run', 'offroad_bringup', 'nav_goal', '22', '1.4', '0', '--timeout', '90'], stdout=subprocess.PIPE, text=True)
t0, nxt, rows = time.time(), 0, []
while goal.poll() is None and time.time() - t0 < 130:
    rclpy.spin_once(n, timeout_sec=0.1)
    if time.time() - t0 >= nxt:
        nxt += 1.0
        rows.append((time.time() - t0, S['state'], S['x'], S['v'], S['wall']))
for t, st, x, v, w in rows:
    print(f'{t:5.1f}s light {st:6s} wall_pts {w:3d} x {x:6.2f} (bumper {x + 3.68:6.2f}) v {v:4.1f}')
print(goal.stdout.read())
