"""Fit the 2D sim's vehicle lag to Webots.

Drives a fixed open-loop command sequence through /cmd_vel, records /odom,
then grid-searches VehicleParams so the 2D model reproduces the speed and
yaw-rate response. Put the car somewhere open first (about 60 m of turning),
or pass --teleport X Y YAW (odom frame).

    ros2 run vesta_rl calibrate --out calib.csv
Then put the printed values into VehicleParams() defaults / random() ranges in sim2d.py.
"""
import argparse
import itertools
import math

import numpy as np

from . import config as C
from .sim2d import Vehicle, VehicleParams

# (duration s, v m/s, delta rad)
SEQUENCE = [(6.0, 3.0, 0.0), (6.0, 3.0, 0.3), (6.0, 3.0, -0.3), (4.0, 1.0, 0.5), (4.0, 0.0, 0.0)]


def commands():
    out = []
    for dur, v, d in SEQUENCE:
        out += [(v, d)] * int(round(dur / C.DT))
    return np.array(out)


def record_webots(cmds, teleport=None):
    import rclpy
    from geometry_msgs.msg import PoseStamped

    from .webots_env import WebotsEnv
    env = WebotsEnv.__new__(WebotsEnv)  # reuse its ROS helpers, skip the path library
    _init_ros(env)
    env._wait_for(lambda: env.odom is not None)
    if teleport:
        x, y, yaw = teleport
        msg = PoseStamped()
        msg.header.frame_id = 'odom'
        msg.pose.position.x, msg.pose.position.y = x, y
        msg.pose.orientation.z, msg.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        env.teleport_pub.publish(msg)
    env._publish(0.0, 0.0)
    env._wait_sim(2.0)
    log = []
    for v, d in cmds:
        env._publish(v, d)
        env._wait_sim(C.DT)
        tw = env.odom.twist.twist
        log.append((tw.linear.x, tw.angular.z))
    env._publish(0.0, 0.0)
    env.node.destroy_node()
    rclpy.shutdown()
    return np.array(log)


def _init_ros(env):
    """WebotsEnv's ROS setup without loading a path library."""
    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Odometry
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock
    if not rclpy.ok():
        rclpy.init()
    env.node = rclpy.create_node('vesta_calibrate', parameter_overrides=[
        rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
    env.cmd_pub = env.node.create_publisher(Twist, '/cmd_vel', QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE))
    env.teleport_pub = env.node.create_publisher(PoseStamped, '/vesta/teleport', 1)
    env.node.create_subscription(Odometry, '/odom', env._on_odom, qos_profile_sensor_data)
    env.node.create_subscription(Clock, '/clock', env._on_clock, 10)
    env.odom, env.now = None, 0.0


def simulate(cmds, params, v_meas=None):
    """Returns (v, yaw_rate) per tick. With v_meas, yaw rate uses the measured speed
    so steering can be fitted independently of the speed response."""
    car = Vehicle(params)
    out = []
    for i, (v, d) in enumerate(cmds):
        car.step(v, d)
        speed = car.v if v_meas is None else v_meas[i]
        out.append((car.v, speed * math.tan(car.delta) / C.WHEELBASE))
    return np.array(out)


def fit(cmds, log):
    v_meas = log[:, 0]
    best_v = min(
        ((np.mean((simulate(cmds, VehicleParams(tau_v=t, accel_max=a))[:, 0] - v_meas) ** 2), t, a)
         for t, a in itertools.product(np.linspace(0.1, 2.0, 20), np.linspace(0.5, 5.0, 19))))
    best_d = min(
        ((np.mean((simulate(cmds, VehicleParams(tau_delta=t, steer_rate=r), v_meas)[:, 1] - log[:, 1]) ** 2), t, r)
         for t, r in itertools.product(np.linspace(0.02, 0.5, 13), np.linspace(0.2, 3.0, 15))))
    return VehicleParams(tau_v=best_v[1], accel_max=best_v[2], tau_delta=best_d[1], steer_rate=best_d[2]), \
        math.sqrt(best_v[0]), math.sqrt(best_d[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='calib.csv')
    ap.add_argument('--from-csv', help='refit an earlier recording instead of driving')
    ap.add_argument('--teleport', type=float, nargs=3, metavar=('X', 'Y', 'YAW'))
    args = ap.parse_args()

    cmds = commands()
    if args.from_csv:
        log = np.loadtxt(args.from_csv, delimiter=',', skiprows=1)[:, 2:4]
    else:
        log = record_webots(cmds, args.teleport)
        np.savetxt(args.out, np.hstack([cmds, log]), delimiter=',', header='v_cmd,delta_cmd,v,yaw_rate', comments='')
        print('recorded', args.out)

    nominal = simulate(cmds, VehicleParams())
    p, rmse_v, rmse_w = fit(cmds, log)
    print(f'nominal 2D model: rmse v {math.sqrt(np.mean((nominal[:, 0] - log[:, 0]) ** 2)):.3f} m/s, '
          f'yaw rate {math.sqrt(np.mean((nominal[:, 1] - log[:, 1]) ** 2)):.3f} rad/s')
    print(f'fitted: VehicleParams(tau_v={p.tau_v:.2f}, accel_max={p.accel_max:.2f}, '
          f'tau_delta={p.tau_delta:.2f}, steer_rate={p.steer_rate:.2f})  '
          f'rmse v {rmse_v:.3f} m/s, yaw rate {rmse_w:.3f} rad/s')


if __name__ == '__main__':
    main()
