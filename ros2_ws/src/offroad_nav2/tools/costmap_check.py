#!/usr/bin/env python3
"""P7 acceptance for the local costmap (run in the sim container).
  halo    : stop perception_node first. Publishes synthetic VRU and vehicle clouds and reports the halo radius of each
            (distance from the lethal core to the last non-free cell): VRU must be larger.
  barrel  : lethal cells vs lidar-measured barrels: for each barrel-like lidar blob within 4-20 m, distance from its
            centre to the nearest lethal cell and (reverse) from lethal cells of the barrel to the blob.
  clear   : stationary car in front of barrels; reseeds the barrels and reports how long the old lethal cells take to
            clear after the lidar no longer sees the barrel.
"""
import math
import sys
import time

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Int32
from tf2_ros import Buffer, TransformListener


class C(Node):
    def __init__(self):
        super().__init__('costmap_check', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', value=True)])
        q = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.grid, self.lidar = None, None
        self.buf = Buffer()
        self.tl = TransformListener(self.buf, self)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', lambda m: setattr(self, 'grid', m), q)
        self.create_subscription(PointCloud2, '/lidar', lambda m: setattr(self, 'lidar', m), qos_profile_sensor_data)
        self.pubs = {k: self.create_publisher(PointCloud2, f'/perception/points/{k}', 5) for k in ('vru', 'vehicle', 'geometric')}
        self.seed = self.create_publisher(Int32, '/vesta/barrels_seed', 1)

    def spin(self, s):
        t = time.time()
        while time.time() - t < s:
            rclpy.spin_once(self, timeout_sec=0.05)

    def costs(self):
        g = self.grid
        a = np.array(g.data, np.int8).reshape(g.info.height, g.info.width)
        return a, g.info.resolution, g.info.origin.position.x, g.info.origin.position.y

    def synth(self, topic, centre, radius, frame='odom'):
        pts = []
        for z in (0.3, 0.8, 1.3):
            for a in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                pts.append((centre[0] + radius * math.cos(a), centre[1] + radius * math.sin(a), z - 0.108))
        pts = np.array(pts, np.float32)
        m = PointCloud2()
        m.header.stamp, m.header.frame_id = self.get_clock().now().to_msg(), frame
        m.height, m.width = 1, len(pts)
        m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate('xyz')]
        m.is_bigendian, m.point_step, m.is_dense, m.row_step = False, 12, True, 12 * len(pts)
        import array
        m.data = array.array('B', pts.tobytes())
        self.pubs[topic].publish(m)

    def lidar_pts_odom(self):
        m = self.lidar
        tf = self.buf.lookup_transform('odom', m.header.frame_id, rclpy.time.Time())
        q, t = tf.transform.rotation, tf.transform.translation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        p = np.array([[a, b, c] for a, b, c in point_cloud2.read_points(m, field_names=('x', 'y', 'z'), skip_nans=True)]).reshape(-1, 3)
        return p @ R.T + [t.x, t.y, t.z]

    def barrels(self):
        """Barrel-like lidar blobs in odom (x, y)."""
        p = self.lidar_pts_odom()
        p = p[(p[:, 2] > -0.108 + 0.3) & (p[:, 2] < -0.108 + 1.2)]
        base = self.buf.lookup_transform('odom', 'base_link', rclpy.time.Time()).transform.translation
        rel = p[:, :2] - [base.x, base.y]
        rel = rel[(abs(rel) < 39).all(1)]
        c = 0.15
        k = np.floor((rel + 40) / c).astype(int)
        g = np.zeros((int(80 / c) + 2,) * 2, np.uint8)
        g[k[:, 0], k[:, 1]] = 1
        n, lab, st, cen = cv2.connectedComponentsWithStats(g, connectivity=8)
        out = []
        for i in range(1, n):
            w, h, a = st[i, cv2.CC_STAT_WIDTH] * c, st[i, cv2.CC_STAT_HEIGHT] * c, st[i, cv2.CC_STAT_AREA]
            if 0.3 < w < 1.2 and 0.3 < h < 1.2 and a >= 4:
                cx, cy = cen[i][1] * c - 40 + base.x, cen[i][0] * c - 40 + base.y
                if 4 < math.hypot(cx - base.x, cy - base.y) < 22:
                    out.append((cx, cy))
        return out


def lethal_dist(a, r, ox, oy, x, y):
    ii, jj = np.nonzero(a >= 99)
    if len(ii) == 0:
        return float('inf')
    return float(np.min(np.hypot(ox + (jj + 0.5) * r - x, oy + (ii + 0.5) * r - y)))


def main():
    mode = sys.argv[1]
    rclpy.init()
    c = C()
    c.spin(3)
    if mode == 'halo':
        a0, r, ox, oy = c.costs()
        base = c.buf.lookup_transform('odom', 'base_link', rclpy.time.Time()).transform
        yaw = 2 * math.atan2(base.rotation.z, base.rotation.w)
        rel = {'vru': (14.0, 6.0), 'vehicle': (14.0, -6.0)}
        odom = {k: (base.translation.x + math.cos(yaw) * v[0] - math.sin(yaw) * v[1],
                    base.translation.y + math.sin(yaw) * v[0] + math.cos(yaw) * v[1]) for k, v in rel.items()}
        for _ in range(30):   # clouds in base_link so the layers' 20 m range gate is measured from the car
            c.synth('vru', rel['vru'], 0.25, 'base_link')
            c.synth('vehicle', rel['vehicle'], 0.9, 'base_link')
            c.spin(0.1)
        a, r, ox, oy = c.costs()
        jj, ii = np.meshgrid(np.arange(a.shape[1]), np.arange(a.shape[0]))
        for name, k in (('vru (cloud radius 0.25 m)', 'vru'), ('vehicle (cloud radius 0.9 m)', 'vehicle')):
            d = np.hypot(ox + (jj + 0.5) * r - odom[k][0], oy + (ii + 0.5) * r - odom[k][1])
            new_cells = (a > 0) & (a0 == 0) & (d < 6)
            lethal = new_cells & (a >= 99)
            print(f'{name}: lethal core radius {d[lethal].max() if lethal.any() else float("nan"):.2f} m, '
                  f'halo (outermost newly non-free cell) {d[new_cells].max() if new_cells.any() else float("nan"):.2f} m')
    elif mode == 'barrel':
        bl = c.barrels()
        a, r, ox, oy = c.costs()
        print(f'{len(bl)} barrel-like lidar blobs between 4 and 22 m')
        base = c.buf.lookup_transform('odom', 'base_link', rclpy.time.Time()).transform
        yaw = 2 * math.atan2(base.rotation.z, base.rotation.w)
        for x, y in bl:
            dx, dy = x - base.translation.x, y - base.translation.y
            rng = math.hypot(dx, dy)
            brg = math.degrees(math.atan2(-math.sin(yaw) * dx + math.cos(yaw) * dy, math.cos(yaw) * dx + math.sin(yaw) * dy))
            seen = abs(brg) < 25 and 4 < rng < 20      # inside the camera's 57 deg field of view and the 20 m depth limit
            print(f'  blob range {rng:5.1f} m bearing {brg:6.1f} deg {"IN VIEW" if seen else "out of view"}: nearest lethal costmap cell {lethal_dist(a, r, ox, oy, x, y):.2f} m away')
    elif mode == 'clear':
        before = c.barrels()
        a, r, ox, oy = c.costs()
        tgt = [(x, y) for x, y in before if lethal_dist(a, r, ox, oy, x, y) < 0.5]
        print(f'{len(tgt)} barrels currently lethal; reseeding')
        c.seed.publish(Int32(data=int(sys.argv[2]) if len(sys.argv) > 2 else 99))
        t0 = time.time()
        gone_t, clear_t = None, None
        while time.time() - t0 < 40 and tgt:
            c.spin(0.25)
            now = c.barrels()
            a, r, ox, oy = c.costs()
            x, y = tgt[0]
            seen = any(math.hypot(x - bx, y - by) < 0.6 for bx, by in now)
            if not seen and gone_t is None:
                gone_t = time.time() - t0
            if lethal_dist(a, r, ox, oy, x, y) > 0.5 and gone_t is not None and clear_t is None:
                clear_t = time.time() - t0
                break
        print(f'barrel gone from lidar at +{gone_t}, cell cleared at +{clear_t} s => clear delay {None if clear_t is None or gone_t is None else round(clear_t - gone_t, 2)} s')


if __name__ == '__main__':
    main()
