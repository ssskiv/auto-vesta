#!/usr/bin/env python3
"""Camera-derived obstacle clouds vs the VLP-16 on a stationary car (Webots ground truth check).
Both in `odom` (BEV). For camera points in a range band: distance to the nearest lidar obstacle point.
Also the recall of lidar obstacle blobs by camera points. Run inside the sim container."""
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformListener
from scipy.spatial import cKDTree

NAMES = sys.argv[1].split(',') if len(sys.argv) > 1 else ['geometric']
SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 8
GROUND_ODOM_Z = -0.108       # base_link height above ground (measured, see vesta.urdf)
H_MIN = 0.2                    # lidar returns at least 20 cm above ground = obstacle


def quat_R(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class E(Node):
    def __init__(self):
        super().__init__('eval_clouds', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', value=True)])
        self.buf = Buffer()
        self.tl = TransformListener(self.buf, self)
        self.cam, self.lid = {n: [] for n in NAMES}, []
        q = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        for n in NAMES:
            self.create_subscription(PointCloud2, f'/perception/points/{n}', lambda m, n=n: self.on(m, self.cam[n]), q)
        self.create_subscription(PointCloud2, '/lidar', lambda m: self.on(m, self.lid), qos_profile_sensor_data)

    def on(self, m, store):
        try:
            # car is stationary: latest TF is fine (a blocking lookup inside a callback would starve the TF listener)
            tf = self.buf.lookup_transform('odom', m.header.frame_id, rclpy.time.Time())
        except Exception:
            return
        pts = np.array([[a, b, c] for a, b, c in point_cloud2.read_points(m, field_names=('x', 'y', 'z'), skip_nans=True)]).reshape(-1, 3)
        if len(pts) == 0:
            return
        t = tf.transform.translation
        store.append(pts @ quat_R(tf.transform.rotation).T + np.array([t.x, t.y, t.z]))


def main():
    rclpy.init()
    n = E()
    t0 = time.time()
    while time.time() - t0 < SECS:
        rclpy.spin_once(n, timeout_sec=0.1)
    lid = np.vstack(n.lid) if n.lid else np.zeros((0, 3))
    lid = lid[lid[:, 2] > GROUND_ODOM_Z + H_MIN]
    base = n.buf.lookup_transform('odom', 'base_link', rclpy.time.Time()).transform.translation
    print(f'lidar obstacle points {len(lid)} ({len(n.lid)} scans)')
    tree = cKDTree(lid[:, :2])
    for name in NAMES:
        if not n.cam[name]:
            print(name, 'no data'); continue
        cam = np.vstack(n.cam[name])
        rng = np.hypot(cam[:, 0] - base.x, cam[:, 1] - base.y)
        d, _ = tree.query(cam[:, :2])
        print(f'[{name}] {len(cam)} points from {len(n.cam[name])} frames; distance of camera points to nearest lidar obstacle point (BEV):')
        print('  range(m)     n   median   p90   frac<0.3m')
        for lo, hi in [(3, 8), (8, 12), (12, 15), (15, 20)]:
            m = (rng >= lo) & (rng < hi)
            if m.sum():
                print(f'  {lo:2d}-{hi:<3d} {m.sum():8d}  {np.median(d[m]):6.2f} {np.percentile(d[m], 90):6.2f}   {100 * (d[m] < 0.3).mean():5.1f}%')
        # recall: lidar blobs seen by camera
        ct = cKDTree(cam[:, :2])
        lr = np.hypot(lid[:, 0] - base.x, lid[:, 1] - base.y)
        dl, _ = ct.query(lid[:, :2])
        print('  lidar obstacle points with a camera point within 0.5 m (recall):')
        for lo, hi in [(3, 8), (8, 12), (12, 15), (15, 20)]:
            m = (lr >= lo) & (lr < hi) & (np.abs(np.arctan2(lid[:, 1] - base.y, lid[:, 0] - base.x)) < 0.6)
            if m.sum():
                print(f'  {lo:2d}-{hi:<3d} {m.sum():8d}  {100 * (dl[m] < 0.5).mean():5.1f}%')


if __name__ == '__main__':
    main()
