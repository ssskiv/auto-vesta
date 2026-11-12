#!/usr/bin/env python3
"""Teleport the Webots car to a MAP pose via /vesta/teleport (pose in the driver's odom frame, antenna based):
odom = map + (2.2, -1.6) - antenna_spawn_world(2.8345, -2.2609) + 0.4*(cos yaw, sin yaw).  Usage: teleport X Y YAW"""
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped

x, y, yaw = map(float, sys.argv[1:4])
rclpy.init()
n = rclpy.create_node('teleport')
pub = n.create_publisher(PoseStamped, '/vesta/teleport', 1)
m = PoseStamped()
m.header.frame_id = 'odom'
m.pose.position.x = x + 2.2 - 2.8345 + 0.4 * math.cos(yaw)
m.pose.position.y = y - 1.6 + 2.2609 + 0.4 * math.sin(yaw)
m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
for _ in range(10):
    pub.publish(m)
    rclpy.spin_once(n, timeout_sec=0.1)
time.sleep(1)
print('teleported to map', x, y, yaw)
