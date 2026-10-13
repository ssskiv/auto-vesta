"""Drive to a goal in Webots through Nav2: planner_server (ComputePathToPose) ->
controller_server FollowPath (vesta_neural_controller). Prints progress and how
much of the drive the neural network was in control.

    ros2 run vesta_rl drive --goal X Y [YAW]       # map frame
    ros2 run vesta_rl drive --library               # end of a library path starting near the car

Needs robot.launch.py and nav2.launch.py running.
"""
import argparse
import collections
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, FollowPath
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from .paths import load_paths

STATUS = {4: 'SUCCEEDED', 5: 'CANCELED', 6: 'ABORTED'}


def pose(x, y, yaw, stamp):
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.header.stamp = stamp
    p.pose.position.x, p.pose.position.y = float(x), float(y)
    p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--goal', type=float, nargs='+', metavar='X Y [YAW]')
    ap.add_argument('--library', action='store_true',
                    help='goal = end of a library path that starts near the car, heading the way the car faces')
    ap.add_argument('--paths', default='src/vesta_rl/data/paths.npz')
    ap.add_argument('--min-length', type=float, default=60.0)
    ap.add_argument('--timeout', type=float, default=300.0, help='wall-clock seconds')
    args = ap.parse_args()

    rclpy.init()
    node = rclpy.create_node('vesta_drive', parameter_overrides=[
        rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
    state = {'odom': None}
    sources = collections.Counter()
    node.create_subscription(Odometry, '/odom', lambda m: state.update(odom=m), qos_profile_sensor_data)
    node.create_subscription(String, '/neural_controller/source', lambda m: sources.update([m.data]), 10)
    while state['odom'] is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    p = state['odom'].pose.pose.position
    x0, y0 = p.x, p.y  # map == odom (static identity map->odom)
    q = state['odom'].pose.pose.orientation
    yaw0 = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    if args.goal:
        gx, gy = args.goal[0], args.goal[1]
        gyaw = args.goal[2] if len(args.goal) > 2 else 0.0
    else:
        best = None
        for xy in load_paths(args.paths):
            length = float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))
            d = math.hypot(xy[0, 0] - x0, xy[0, 1] - y0)
            h = math.atan2(xy[6, 1] - xy[0, 1], xy[6, 0] - xy[0, 0])
            ahead = abs((h - yaw0 + math.pi) % (2 * math.pi) - math.pi) < math.radians(45)
            if length >= args.min_length and ahead and (best is None or d < best[0]):
                best = (d, xy)
        if best is None:
            raise SystemExit('no library path long enough')
        xy = best[1]
        gx, gy = xy[-1]
        gyaw = math.atan2(xy[-1, 1] - xy[-3, 1], xy[-1, 0] - xy[-3, 0])
        print(f'library path starts {best[0]:.1f} m from the car')
    print(f'car at ({x0:.1f}, {y0:.1f}, yaw {yaw0:.2f}), goal ({gx:.1f}, {gy:.1f}, yaw {gyaw:.2f})', flush=True)

    planner = ActionClient(node, ComputePathToPose, 'compute_path_to_pose')
    follower = ActionClient(node, FollowPath, 'follow_path')
    assert planner.wait_for_server(timeout_sec=20.0), 'planner_server not up (nav2.launch.py?)'
    assert follower.wait_for_server(timeout_sec=20.0), 'controller_server not up'

    stamp = node.get_clock().now().to_msg()
    fut = planner.send_goal_async(ComputePathToPose.Goal(goal=pose(gx, gy, gyaw, stamp), planner_id='GridBased'))
    rclpy.spin_until_future_complete(node, fut)
    res = fut.result().get_result_async()
    rclpy.spin_until_future_complete(node, res, timeout_sec=30.0)
    path = res.result().result.path if res.done() else None
    if not path or not path.poses:
        raise SystemExit('planner returned no path')
    pts = np.array([[q.pose.position.x, q.pose.position.y] for q in path.poses])
    plen = float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))
    print(f'path: {len(pts)} poses, {plen:.1f} m', flush=True)

    fut = follower.send_goal_async(FollowPath.Goal(path=path, controller_id='FollowPath',
                                                   goal_checker_id='general_goal_checker'))
    rclpy.spin_until_future_complete(node, fut)
    handle = fut.result()
    if not handle.accepted:
        raise SystemExit('FollowPath rejected')
    res = handle.get_result_async()
    t0 = time.monotonic()
    sim0 = node.get_clock().now().nanoseconds * 1e-9
    last = 0.0
    while not res.done() and time.monotonic() - t0 < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.monotonic() - last > 3.0:
            last = time.monotonic()
            o = state['odom']
            q = o.pose.pose.position
            dist_goal = math.hypot(q.x - gx, q.y - gy)
            n = sum(sources.values()) or 1
            print(f'  t={node.get_clock().now().nanoseconds * 1e-9 - sim0:5.1f}s  pos ({q.x:6.1f}, {q.y:6.1f})  '
                  f'v {o.twist.twist.linear.x:4.2f} m/s  to goal {dist_goal:5.1f} m  '
                  f'NN {sources["NN"] / n:.0%} RPP {sources["RPP"] / n:.0%} STOP {sources["STOP"] / n:.0%}', flush=True)
    if not res.done():
        handle.cancel_goal_async()
        rclpy.spin_once(node, timeout_sec=0.5)
        status = 'TIMEOUT'
    else:
        status = STATUS.get(res.result().status, str(res.result().status))
    n = sum(sources.values()) or 1
    q = state['odom'].pose.pose.position
    print(f'RESULT {status}: sim time {node.get_clock().now().nanoseconds * 1e-9 - sim0:.1f} s, '
          f'{math.hypot(q.x - gx, q.y - gy):.1f} m from goal, control share: '
          f'NN {sources["NN"] / n:.0%}, RPP {sources["RPP"] / n:.0%}, STOP {sources["STOP"] / n:.0%} ({n} ticks)', flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
