"""GPS waypoint mission: lat/lon list -> map poses via navsat_transform's /fromLL -> Nav2 FollowWaypoints.
Humble's Nav2 has no FollowGPSWaypoints. Usage: gps_route lat,lon lat,lon ...   (or --map x,y x,y ... to build the
list from map coordinates with the sim datum, for tests). Keep spacing between waypoints < 100 m."""
import math
import sys
import time

import rclpy
from geographic_msgs.msg import GeoPoint
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import FollowWaypoints
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from robot_localization.srv import FromLL

R_EARTH = 6378137.0
DATUM = (54.3282, 48.3866)   # = sim_sensor_adapter / navsat.yaml datum. TODO(user): real venue


def map_to_ll(x, y):
    return (DATUM[0] + math.degrees(y / R_EARTH), DATUM[1] + math.degrees(x / (R_EARTH * math.cos(math.radians(DATUM[0])))))


def main():
    a = sys.argv[1:]
    pts = []
    if a and a[0] == '--map':
        pts = [map_to_ll(*map(float, t.split(','))) for t in a[1:]]
    else:
        pts = [tuple(map(float, t.split(','))) for t in a]
    rclpy.init()
    n = Node('gps_route', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', value=True)])
    st = {'neg': 0, 'vmax': 0.0, 'n': 0}
    n.create_subscription(Twist, '/cmd_vel', lambda m: st.update(neg=st['neg'] + (m.linear.x < -1e-6), n=st['n'] + 1,
                                                                 vmax=max(st['vmax'], m.linear.x)), 50)
    cli = n.create_client(FromLL, '/fromLL')
    if not cli.wait_for_service(timeout_sec=20):
        print('no /fromLL service'); return
    poses = []
    for lat, lon in pts:
        r = FromLL.Request()
        r.ll_point = GeoPoint(latitude=lat, longitude=lon, altitude=0.0)
        f = cli.call_async(r)
        rclpy.spin_until_future_complete(n, f, timeout_sec=5)
        p = f.result().map_point
        poses.append((p.x, p.y))
        print(f'lat {lat:.6f} lon {lon:.6f} -> map ({p.x:.2f}, {p.y:.2f})')
    goal = FollowWaypoints.Goal()
    for i, (x, y) in enumerate(poses):
        nx, ny = poses[min(i + 1, len(poses) - 1)] if i + 1 < len(poses) else (x + (x - poses[i - 1][0]), y + (y - poses[i - 1][1])) if i else (x + 1, y)
        yaw = math.atan2(ny - y, nx - x)
        ps = PoseStamped()
        ps.header.frame_id = 'map'
        ps.pose.position.x, ps.pose.position.y = x, y
        ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        goal.poses.append(ps)
    ac = ActionClient(n, FollowWaypoints, 'follow_waypoints')
    ac.wait_for_server(timeout_sec=20)
    t0 = time.time()
    fut = ac.send_goal_async(goal)
    rclpy.spin_until_future_complete(n, fut, timeout_sec=10)
    gh = fut.result()
    rf = gh.get_result_async()
    while not rf.done() and time.time() - t0 < 600:
        rclpy.spin_once(n, timeout_sec=0.2)
    res = rf.result().result if rf.done() else None
    print(f'FollowWaypoints done={rf.done()} missed={list(res.missed_waypoints) if res else "?"} wall {time.time() - t0:.0f}s')
    print(f'/cmd_vel: {st["n"]} msgs, max linear.x {st["vmax"]:.2f} m/s ({st["vmax"] * 3.6:.1f} km/h), negative msgs: {st["neg"]}')


if __name__ == '__main__':
    main()
