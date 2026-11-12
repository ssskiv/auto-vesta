"""Drive one lap of the RoboCross track with FollowWaypoints (route: config/lap_route.json, map frame) and print the
referee result (/tmp/referee_results.json, written by the Webots referee in the same container).
Usage: lap_run [--timeout S] [--out FILE]"""
import json
import math
import os
import sys
import time

import rclpy
from ament_index_python.packages import get_package_share_directory as share
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import FollowWaypoints
from rclpy.action import ActionClient
from rclpy.node import Node


def main():
    a = sys.argv[1:]
    timeout = float(a[a.index('--timeout') + 1]) if '--timeout' in a else 600.0
    out = a[a.index('--out') + 1] if '--out' in a else None
    full = json.load(open(os.path.join(share('offroad_nav2'), 'config', 'lap_route.json')))
    route = full[1:] + [dict(full[0], yaw=0.0)]    # close the loop: finish in the start-finish zone
    if '--from' in a:   # debugging: start from this waypoint index (after teleporting the car there)
        route = route[int(a[a.index('--from') + 1]):]
    rclpy.init()
    n = Node('lap_run', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', value=True)])
    st = {'neg': 0, 'vmax': 0.0, 'n': 0}
    n.create_subscription(Twist, '/cmd_vel', lambda m: st.update(neg=st['neg'] + (m.linear.x < -1e-6), n=st['n'] + 1,
                                                                 vmax=max(st['vmax'], m.linear.x)), 50)
    ac = ActionClient(n, FollowWaypoints, 'follow_waypoints')
    ac.wait_for_server(timeout_sec=30)
    g = FollowWaypoints.Goal()
    for w in route:
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x, p.pose.position.y = w['x'], w['y']
        p.pose.orientation.z, p.pose.orientation.w = math.sin(w['yaw'] / 2), math.cos(w['yaw'] / 2)
        g.poses.append(p)
    t0 = time.time()
    gh = None
    for attempt in range(5):   # the accept response can be lost during DDS discovery: retry
        f = ac.send_goal_async(g)
        rclpy.spin_until_future_complete(n, f, timeout_sec=20)
        gh = f.result()
        if gh is not None and gh.accepted:
            break
        print(f'goal not accepted (attempt {attempt + 1}), retrying')
        time.sleep(3)
    rf = gh.get_result_async()
    while not rf.done() and time.time() - t0 < timeout:
        rclpy.spin_once(n, timeout_sec=0.2)
    done = rf.done()
    missed = list(rf.result().result.missed_waypoints) if done else None
    res = {'lap_done': done, 'missed_waypoints': missed, 'wall_s': round(time.time() - t0),
           'cmd_vel_msgs': st['n'], 'max_linear_x': round(st['vmax'], 3), 'negative_linear_x_msgs': st['neg']}
    try:
        res['referee'] = json.load(open(os.environ.get('REFEREE_OUT', '/tmp/referee_results.json')))
    except Exception as e:
        res['referee'] = f'unavailable: {e}'
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if out:
        json.dump(res, open(out, 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
