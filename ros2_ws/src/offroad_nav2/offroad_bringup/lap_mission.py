"""Lap mission with live waypoint adjustment: NavigateToPose to each route waypoint in turn. Before sending, the
waypoint is snapped to the nearest free spot of the live global costmap (a barrel on the centreline would otherwise make the
planner fail), and a goal that fails or times out is skipped (a forward-only car cannot back up and retry).
Usage: lap_mission [--timeout S] [--out FILE] [--from N]"""
import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory as share
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


class Mission(Node):
    def __init__(self):
        super().__init__('lap_mission', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', value=True)])
        q = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.grid, self.pos = None, None
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', lambda m: setattr(self, 'grid', m), q)
        self.create_subscription(Odometry, '/odometry/global', lambda m: setattr(self, 'pos', (m.pose.pose.position.x, m.pose.pose.position.y)), 5)
        self.st = {'neg': 0, 'vmax': 0.0, 'n': 0}
        self.create_subscription(Twist, '/cmd_vel', lambda m: self.st.update(neg=self.st['neg'] + (m.linear.x < -1e-6), n=self.st['n'] + 1, vmax=max(self.st['vmax'], m.linear.x)), 50)
        self.ac = ActionClient(self, NavigateToPose, 'navigate_to_pose')

    def snap(self, x, y, radius=5.0, clear=1.6):
        """Nearest cell to (x, y) whose neighbourhood of `clear` m is free in the global costmap."""
        g = self.grid
        if g is None:
            return x, y
        a = np.array(g.data, np.int8).reshape(g.info.height, g.info.width)
        r, ox, oy = g.info.resolution, g.info.origin.position.x, g.info.origin.position.y
        ci, cj = int((y - oy) / r), int((x - ox) / r)
        k = int(radius / r)
        sub = a[max(ci - k, 0):ci + k + 1, max(cj - k, 0):cj + k + 1]
        bad = (sub < 0) | (sub >= 60)
        import cv2
        free_dist = cv2.distanceTransform((~bad).astype(np.uint8), cv2.DIST_L2, 3) * r
        ii, jj = np.nonzero(free_dist >= clear)
        if len(ii) == 0:
            return x, y
        y0, x0 = max(ci - k, 0), max(cj - k, 0)
        d = np.hypot((ii + y0 - ci) * r, (jj + x0 - cj) * r)
        m = int(np.argmin(d))
        return ox + (jj[m] + x0 + 0.5) * r, oy + (ii[m] + y0 + 0.5) * r

    def go(self, x, y, yaw, timeout):
        g = NavigateToPose.Goal()
        g.pose = PoseStamped()
        g.pose.header.frame_id = 'map'
        g.pose.pose.position.x, g.pose.pose.position.y = x, y
        g.pose.pose.orientation.z, g.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        t0 = time.time()
        gh = None
        for _ in range(4):
            f = self.ac.send_goal_async(g)
            rclpy.spin_until_future_complete(self, f, timeout_sec=15)
            gh = f.result()
            if gh is not None and gh.accepted:
                break
            time.sleep(2)
        if gh is None or not gh.accepted:
            return 'rejected'
        rf = gh.get_result_async()
        while not rf.done() and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.2)
        if not rf.done():
            gh.cancel_goal_async()
            return 'timeout'
        return {4: 'ok', 5: 'canceled', 6: 'aborted'}.get(rf.result().status, '?')


def main():
    a = sys.argv[1:]
    timeout = float(a[a.index('--timeout') + 1]) if '--timeout' in a else 900.0
    out = a[a.index('--out') + 1] if '--out' in a else None
    full = json.load(open(os.path.join(share('offroad_nav2'), 'config', 'lap_route.json')))
    route = full[1:] + [dict(full[0], yaw=0.0)]
    if '--from' in a:
        route = route[int(a[a.index('--from') + 1]):]
    rclpy.init()
    m = Mission()
    m.ac.wait_for_server(timeout_sec=30)
    for _ in range(50):
        rclpy.spin_once(m, timeout_sec=0.1)
    t0 = time.time()
    log = []
    for i, w in enumerate(route):
        if time.time() - t0 > timeout:
            log.append((i, 'mission timeout')); break
        x, y = m.snap(w['x'], w['y'])
        res = m.go(x, y, w['yaw'], 60.0)
        log.append((i, f'({x:.1f},{y:.1f}) shifted {math.hypot(x - w["x"], y - w["y"]):.1f} m: {res}'))
        print(log[-1], flush=True)
    res = {'wall_s': round(time.time() - t0), 'goals': [f'{i}: {s}' for i, s in log],
           'goals_failed': sum(1 for _, s in log if not s.endswith('ok')),
           'cmd_vel_msgs': m.st['n'], 'max_linear_x': round(m.st['vmax'], 3), 'negative_linear_x_msgs': m.st['neg']}
    try:
        res['referee'] = json.load(open(os.environ.get('REFEREE_OUT', '/tmp/referee_results.json')))
    except Exception as e:
        res['referee'] = f'unavailable: {e}'
    res['lap_done'] = True
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if out:
        json.dump(res, open(out, 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
