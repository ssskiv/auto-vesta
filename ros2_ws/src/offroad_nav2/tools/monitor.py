#!/usr/bin/env python3
"""Diagnostics during a lap: every 3 s log pose, speed, global-costmap cost at the robot and lethal cells within 6 m;
dump a global costmap crop PNG when the car has been stationary > 8 s or stands in a lethal cell."""
import sys
import time

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp'
rclpy.init()
n = Node('monitor')
S = {'g': None, 'odom': None}
q = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
n.create_subscription(OccupancyGrid, '/global_costmap/costmap', lambda m: S.__setitem__('g', m), q)
n.create_subscription(Odometry, '/odometry/global', lambda m: S.__setitem__('odom', m), 5)
t0, last_move, shots, nxt = time.time(), time.time(), 0, 0
prev = None
while rclpy.ok():
    rclpy.spin_once(n, timeout_sec=0.2)
    if time.time() < nxt or S['g'] is None or S['odom'] is None:
        continue
    nxt = time.time() + 3
    g, o = S['g'], S['odom']
    a = np.array(g.data, np.int8).reshape(g.info.height, g.info.width)
    r, ox, oy = g.info.resolution, g.info.origin.position.x, g.info.origin.position.y
    x, y = o.pose.pose.position.x, o.pose.pose.position.y
    ci, cj = int((y - oy) / r), int((x - ox) / r)
    win = a[max(ci - 30, 0):ci + 31, max(cj - 30, 0):cj + 31]
    v = o.twist.twist.linear.x
    if abs(v) > 0.2:
        last_move = time.time()
    stuck = time.time() - last_move > 8
    print(f'{time.time() - t0:5.0f}s pos ({x:6.1f},{y:6.1f}) v {v:4.1f} cost@robot {a[ci, cj]:3d} lethal<6m {(win >= 99).sum():4d}', flush=True)
    if (stuck or a[ci, cj] >= 99) and shots < 4 and (prev is None or time.time() - prev > 20):
        prev = time.time(); shots += 1
        crop = a[max(ci - 100, 0):ci + 101, max(cj - 100, 0):cj + 101]
        img = np.zeros((*crop.shape, 3), np.uint8)
        img[crop == 0] = (255, 255, 255); img[(crop > 0) & (crop < 99)] = (0, 180, 255); img[crop >= 99] = (0, 0, 255); img[crop < 0] = (120, 120, 120)
        img = cv2.flip(img, 0)
        cv2.circle(img, (min(100, cj), crop.shape[0] - 1 - min(100, ci)), 5, (0, 200, 0), -1)
        cv2.imwrite(f'{OUT}/stuck_{shots}.png', cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
        print(f'  -> snapshot stuck_{shots}.png (stuck={stuck})', flush=True)
