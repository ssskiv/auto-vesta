#!/usr/bin/env python3
"""Offline route check (no simulator): with navigation.launch.py + a static map->base_link TF running, ask the
planner (ComputePathToPose) for the path between consecutive lap waypoints. Prints success and path length per leg."""
import json
import math
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient
from rclpy.node import Node

route = json.load(open(sys.argv[1]))
rclpy.init()
n = Node('plan_check')
ac = ActionClient(n, ComputePathToPose, 'compute_path_to_pose')
ac.wait_for_server(timeout_sec=30)


def pose(w):
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.pose.position.x, p.pose.position.y = w['x'], w['y']
    p.pose.orientation.z, p.pose.orientation.w = math.sin(w['yaw'] / 2), math.cos(w['yaw'] / 2)
    return p


bad = 0
for i in range(len(route)):
    a, b = route[i], route[(i + 1) % len(route)]
    g = ComputePathToPose.Goal()
    g.start, g.goal, g.use_start = pose(a), pose(b), True
    g.planner_id = 'GridBased'
    f = ac.send_goal_async(g)
    rclpy.spin_until_future_complete(n, f, timeout_sec=10)
    r = f.result().get_result_async()
    rclpy.spin_until_future_complete(n, r, timeout_sec=15)
    res = r.result()
    ok = res is not None and res.status == 4 and len(res.result.path.poses) > 0
    L = 0.0
    if ok:
        P = res.result.path.poses
        L = sum(math.hypot(P[k + 1].pose.position.x - P[k].pose.position.x, P[k + 1].pose.position.y - P[k].pose.position.y) for k in range(len(P) - 1))
    bad += not ok
    print(f'leg {i:2d} ({a["x"]:6.1f},{a["y"]:6.1f}) -> ({b["x"]:6.1f},{b["y"]:6.1f}): {"OK  %.1f m" % L if ok else "FAILED"}')
print('failed legs:', bad)
