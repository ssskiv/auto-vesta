"""Send NavigateToPose goals (map frame) and report: result, time, distance, /cmd_vel stats (min linear.x must be >= 0),
max speed (from /odometry/local), referee-independent. Usage: nav_goal x y [yaw] [--timeout S] [--then x y ...]"""
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node


class Runner(Node):
    def __init__(self):
        super().__init__('nav_goal', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', value=True)])
        self.ac = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.vmin, self.vmax, self.n, self.neg = 1e9, -1e9, 0, 0
        self.speed_max, self.last, self.dist = 0.0, None, 0.0
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 50)
        self.create_subscription(Odometry, '/odometry/global', self.on_odom, 20)

    def on_cmd(self, m):
        self.n += 1
        self.vmin, self.vmax = min(self.vmin, m.linear.x), max(self.vmax, m.linear.x)
        self.neg += m.linear.x < -1e-6

    def on_odom(self, m):
        p = (m.pose.pose.position.x, m.pose.pose.position.y)
        if self.last:
            self.dist += math.hypot(p[0] - self.last[0], p[1] - self.last[1])
        self.last = p
        self.speed_max = max(self.speed_max, abs(m.twist.twist.linear.x))

    def go(self, x, y, yaw, timeout):
        if not self.ac.wait_for_server(timeout_sec=20):
            return 'no action server'
        g = NavigateToPose.Goal()
        g.pose = PoseStamped()
        g.pose.header.frame_id = 'map'
        g.pose.pose.position.x, g.pose.pose.position.y = x, y
        g.pose.pose.orientation.z, g.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        t0 = time.time()
        fut = self.ac.send_goal_async(g)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10)
        gh = fut.result()
        if gh is None or not gh.accepted:
            return 'goal rejected'
        rf = gh.get_result_async()
        while not rf.done() and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.2)
        if not rf.done():
            gh.cancel_goal_async()
            return f'TIMEOUT after {timeout:.0f}s'
        return {4: 'SUCCEEDED', 5: 'CANCELED', 6: 'ABORTED'}.get(rf.result().status, str(rf.result().status))


def main():
    a = sys.argv[1:]
    timeout = 240.0
    if '--timeout' in a:
        i = a.index('--timeout'); timeout = float(a[i + 1]); del a[i:i + 2]
    goals = []
    cur = []
    for t in a:
        if t == '--then':
            goals.append(cur); cur = []
        else:
            cur.append(float(t))
    goals.append(cur)
    rclpy.init()
    r = Runner()
    t0 = time.time()
    for g in goals:
        x, y = g[0], g[1]
        yaw = g[2] if len(g) > 2 else 0.0
        res = r.go(x, y, yaw, timeout)
        print(f'goal ({x:.1f}, {y:.1f}) -> {res}')
        if res != 'SUCCEEDED':
            break
    print(f'wall time {time.time() - t0:.0f}s, driven {r.dist:.1f} m (global odom)')
    print(f'/cmd_vel: {r.n} msgs, linear.x min {r.vmin:.3f} max {r.vmax:.3f} m/s ({r.vmax * 3.6:.1f} km/h), negative msgs: {r.neg}')
    print(f'max |speed| from /odometry/global: {r.speed_max:.2f} m/s ({r.speed_max * 3.6:.1f} km/h)')


if __name__ == '__main__':
    main()
