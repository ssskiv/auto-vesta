#!/usr/bin/env python3
"""Domain-gap check: teleport to N points along the lap route, save the rectified camera image (1280x720) and the
perception_node segmentation overlay (green = drivable; yellow man-made, red VRU, blue vehicle, magenta natural) for each.
Usage: save_frames.py OUT_DIR [N]"""
import json
import math
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory as share
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

out = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
os.makedirs(out, exist_ok=True)
route = json.load(open(os.path.join(share('offroad_nav2'), 'config', 'lap_route.json')))
rclpy.init()
n = Node('save_frames', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', value=True)])
S = {}
q = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
n.create_subscription(Image, '/stereo/left/image_rect', lambda m: S.__setitem__('rgb', m), q)
n.create_subscription(Image, '/perception/seg_vis', lambda m: S.__setitem__('seg', m), q)
tele = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'teleport.py')
idx = np.linspace(0, len(route) - 1, N).astype(int)
stamps = []
for k, i in enumerate(idx):
    w = route[i]
    subprocess.run([sys.executable, tele, str(w['x']), str(w['y']), str(w['yaw'])], capture_output=True)
    S.clear()
    t0 = time.time()
    while time.time() - t0 < 9 and not ('seg' in S and time.time() - t0 > 5):
        rclpy.spin_once(n, timeout_sec=0.1)
    if 'rgb' not in S or 'seg' not in S:
        print(f'{k}: no frame'); continue
    m = S['rgb']
    img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width * 4].reshape(m.height, m.width, 4)[..., :3]
    cv2.imwrite(f'{out}/{k:02d}_rgb.png', np.ascontiguousarray(img))
    v = S['seg']
    seg = np.frombuffer(v.data, np.uint8).reshape(v.height, v.width, 3)
    cv2.imwrite(f'{out}/{k:02d}_seg.png', cv2.cvtColor(np.ascontiguousarray(seg), cv2.COLOR_RGB2BGR))
    stamps.append((k, m.width, m.height, m.header.stamp.sec + m.header.stamp.nanosec * 1e-9))
    print(f'{k:2d} route wp {i}: rgb {m.width}x{m.height} stamp {stamps[-1][3]:.2f} (sim time)')
