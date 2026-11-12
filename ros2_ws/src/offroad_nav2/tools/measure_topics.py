#!/usr/bin/env python3
"""In-container: rates of the key topics and the image-to-cloud latency (/perception/latency_ms) over a window.
Usage: measure_topics.py [--duration 40] [--json out.json]"""
import argparse
import collections
import json
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32

ap = argparse.ArgumentParser()
ap.add_argument('--duration', type=float, default=40)
ap.add_argument('--json')
a = ap.parse_args()
rclpy.init()
n = rclpy.create_node('measure_topics')
C, lat = collections.Counter(), []
q = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
for t, ty in (('/stereo/left/image_rect', Image), ('/stereo/depth', Image), ('/perception/points/geometric', PointCloud2),
              ('/perception/points/clear', PointCloud2), ('/odometry/global', Odometry), ('/cmd_vel', Twist)):
    n.create_subscription(ty, t, lambda m, t=t: C.update([t]), q)
n.create_subscription(Float32, '/perception/latency_ms', lambda m: lat.append(m.data), 10)
t0 = time.time()
while time.time() - t0 < a.duration:
    rclpy.spin_once(n, timeout_sec=0.05)
res = {'rates_hz': {k: round(v / a.duration, 2) for k, v in C.items()}}
if lat:
    l = np.array(lat)
    res['latency_ms'] = {'n': len(l), 'p50': round(float(np.percentile(l, 50)), 1), 'p95': round(float(np.percentile(l, 95)), 1),
                         'max': round(float(l.max()), 1)}
for k, v in res['rates_hz'].items():
    print(f'{k:36s} {v:6.2f} Hz')
print('latency_ms', res.get('latency_ms'))
if a.json:
    json.dump(res, open(a.json, 'w'), indent=1)
