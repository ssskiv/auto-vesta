#!/usr/bin/env python3
"""Read a rosbag2 of /cmd_vel and report min/max linear.x and the count of negative values."""
import sys

import rosbag2_py
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import Twist

r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=sys.argv[1], storage_id=''), rosbag2_py.ConverterOptions('', ''))
n, neg, lo, hi = 0, 0, 1e9, -1e9
while r.has_next():
    topic, data, _ = r.read_next()
    if topic == '/cmd_vel':
        v = deserialize_message(data, Twist).linear.x
        n += 1; neg += v < -1e-6; lo = min(lo, v); hi = max(hi, v)
print(f'bag {sys.argv[1]}: /cmd_vel msgs {n}, linear.x min {lo:.3f} max {hi:.3f} m/s ({hi * 3.6:.1f} km/h), negative: {neg}')
