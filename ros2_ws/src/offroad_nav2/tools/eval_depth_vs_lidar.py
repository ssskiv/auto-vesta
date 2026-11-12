#!/usr/bin/env python3
"""Compare /stereo/depth with the VLP-16 (/lidar) on a stationary car: project lidar returns into the left
optical frame, sample depth there, report error per distance band.  Run inside the sim container."""
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, __file__.rsplit('/tools/', 1)[0])
from offroad_perception import geometry as G  # noqa: E402


class Eval(Node):
    def __init__(self):
        super().__init__('eval_depth', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', value=True)])
        self.depth, self.cloud, self.K = [], None, None
        self.buf = Buffer()
        self.tl = TransformListener(self.buf, self)
        q = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Image, '/stereo/depth', lambda m: self.depth.append(m) or self.depth.__delitem__(slice(0, -3)), q)
        self.create_subscription(PointCloud2, '/lidar', lambda m: setattr(self, 'cloud', m), qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/stereo/left/camera_info', lambda m: setattr(self, 'K', np.array(m.k).reshape(3, 3)), q)


def main():
    rclpy.init()
    n = Eval()
    t0 = time.time()
    while time.time() - t0 < 12 and (not n.depth or n.cloud is None or n.K is None):
        rclpy.spin_once(n, timeout_sec=0.2)
    t0 = time.time()
    while time.time() - t0 < 3:
        rclpy.spin_once(n, timeout_sec=0.2)
    dm = n.depth[-1]
    depth = np.frombuffer(dm.data, np.float32).reshape(dm.height, dm.width)
    pts = np.array([[p[0], p[1], p[2]] for p in point_cloud2.read_points(n.cloud, field_names=('x', 'y', 'z'), skip_nans=True)])
    tf = n.buf.lookup_transform(dm.header.frame_id, n.cloud.header.frame_id, rclpy.time.Time())
    t, r = tf.transform.translation, tf.transform.rotation
    x, y, z, w = r.x, r.y, r.z, r.w
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    pc = pts @ R.T + np.array([t.x, t.y, t.z])
    pc = pc[pc[:, 2] > 1.0]
    u, v = G.project(pc, n.K)
    ok = (u >= 0) & (u < dm.width - 1) & (v >= 0) & (v < dm.height - 1)
    pc, u, v = pc[ok], u[ok], v[ok]
    d = G.median_depth(depth, u, v, 5)
    print(f'lidar returns in view: {len(pc)}; with valid stereo depth: {np.isfinite(d).sum()}')
    print('band(m)   n    valid%  median_err(m)  p90_abs_err(m)  median_rel_err%')
    for lo, hi in [(2, 5), (5, 10), (10, 15), (15, 20), (20, 30)]:
        m = (pc[:, 2] >= lo) & (pc[:, 2] < hi)
        if m.sum() == 0:
            continue
        e = d[m] - pc[m, 2]
        ev = e[np.isfinite(e)]
        if len(ev) == 0:
            print(f'{lo:2d}-{hi:<3d} {m.sum():6d}   0.0'); continue
        print(f'{lo:2d}-{hi:<3d} {m.sum():6d}  {100*len(ev)/m.sum():5.1f}   {np.median(ev):+7.3f}   {np.percentile(np.abs(ev), 90):7.3f}   {100*np.median(np.abs(ev)/pc[m, 2][np.isfinite(e)]):6.2f}')
    n.destroy_node()


if __name__ == '__main__':
    main()
