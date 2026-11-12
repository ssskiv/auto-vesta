#!/usr/bin/env python3
"""Costmap check (run in the sim container). `halo`: stop perception_node first; publishes synthetic VRU and vehicle clouds and
reports the halo radius of each. (The lidar-based barrel/clear checks were removed with the lidar.)
"""
import math
import sys
import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformListener


class C(Node):
    def __init__(self):
        super().__init__('costmap_check', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', value=True)])
        q = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.grid = None
        self.buf = Buffer()
        self.tl = TransformListener(self.buf, self)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', lambda m: setattr(self, 'grid', m), q)
        self.pubs = {k: self.create_publisher(PointCloud2, f'/perception/points/{k}', 5) for k in ('vru', 'vehicle', 'geometric')}

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


if __name__ == '__main__':
    main()
