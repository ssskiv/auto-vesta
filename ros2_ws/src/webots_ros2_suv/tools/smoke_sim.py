#!/usr/bin/env python3
"""Check a running robot.launch.py; --gate also exercises resets and driving."""
import argparse
import math
from pathlib import Path
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, Imu, JointState, PointCloud2
from std_msgs.msg import Int32
from tf2_msgs.msg import TFMessage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', action='store_true')
    parser.add_argument('--log', type=Path, help='launch log, required for --gate')
    args = parser.parse_args()
    if args.gate and args.log is None:
        parser.error('--gate requires --log')
    rclpy.init()
    node = rclpy.create_node('sim_acceptance')
    latest = {}
    topics = {
        '/clock': Clock, '/odom': Odometry, '/imu/data': Imu,
        '/radar': PointCloud2, '/lidar': PointCloud2,
        '/joint_states': JointState, '/tf': TFMessage,
        '/camera/image_raw/image_color': Image,
        '/stereo_raw/left/image_color': Image,
        '/stereo_raw/right/image_color': Image,
        '/stereo_raw/left/camera_info': CameraInfo,
        '/stereo_raw/right/camera_info': CameraInfo,
    }
    subscriptions = [node.create_subscription(
        kind, topic, lambda msg, topic=topic: latest.__setitem__(topic, msg),
        qos_profile_sensor_data) for topic, kind in topics.items()]
    cmd = node.create_publisher(Twist, '/cmd_vel', 10)
    teleport = node.create_publisher(PoseStamped, '/vesta/teleport', 1)
    seeds = node.create_publisher(Int32, '/vesta/barrels_seed', 1)

    def wait_for(predicate, timeout=60, action=None):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if action:
                action()
            rclpy.spin_once(node, timeout_sec=0.1)
            if predicate():
                return
        raise AssertionError(f'timeout; received topics: {sorted(latest)}')

    def sim_time():
        clock = latest['/clock'].clock
        return clock.sec + clock.nanosec * 1e-9

    try:
        wait_for(lambda: len(latest) == len(topics))
        start = sim_time()
        wait_for(lambda: sim_time() > start + 1)
        for topic in ('/camera/image_raw/image_color',
                      '/stereo_raw/left/image_color', '/stereo_raw/right/image_color'):
            msg = latest[topic]
            assert (msg.width, msg.height) == (1280, 720), topic
            assert msg.data and any(msg.data), f'empty/black image: {topic}'
        for side in ('left', 'right'):
            msg = latest[f'/stereo_raw/{side}/camera_info']
            assert msg.header.frame_id == f'stereo_{side}_optical_frame'
            assert msg.k[0] > 0 and msg.k[4] > 0
        assert latest['/lidar'].width * latest['/lidar'].height > 0
        assert any(t.header.frame_id == 'odom' and t.child_frame_id == 'base_link'
                   for t in latest['/tf'].transforms)
        for topic in ('/odom', '/imu/data', '/radar', '/joint_states'):
            stamp = latest[topic].header.stamp
            assert abs(stamp.sec + stamp.nanosec * 1e-9 - sim_time()) < 2, topic
        print('SMOKE PASS: clock, odometry, TF, IMU, lidar, radar, RGB and stereo', flush=True)
        if not args.gate:
            return
        for seed in (7, 19, 7):
            previous = args.log.read_text().count(f'barrels: new layout, randomSeed {seed}')
            seeds.publish(Int32(data=seed))
            wait_for(lambda: args.log.read_text().count(
                f'barrels: new layout, randomSeed {seed}') > previous)
            start = sim_time()
            wait_for(lambda: sim_time() > start + 1)
        pose = PoseStamped()
        pose.header.frame_id = 'odom'
        pose.pose.position.x, pose.pose.position.y = 5.0, 0.0
        pose.pose.orientation.w = 1.0
        teleport.publish(pose)
        wait_for(lambda: abs(latest['/odom'].pose.pose.position.x - 5) < 0.5
                 and abs(latest['/odom'].pose.pose.position.y) < 0.5)
        for velocity in (1.0, -1.0):
            before = latest['/odom'].pose.pose.position.x
            twist = Twist()
            twist.linear.x = velocity
            wait_for(lambda: (latest['/odom'].pose.pose.position.x - before) * velocity > 0.5,
                     action=lambda: cmd.publish(twist))
            cmd.publish(Twist())
        position = latest['/odom'].pose.pose.position
        assert all(math.isfinite(v) for v in (position.x, position.y, position.z))
        assert abs(position.z) < 1, 'vehicle fell through the track'
        print('GATE PASS: repeated barrel reseed, teleport, forward/reverse, stable physics', flush=True)
    finally:
        cmd.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
