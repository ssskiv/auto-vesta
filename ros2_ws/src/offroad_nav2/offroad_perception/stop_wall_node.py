"""Virtual stop wall at the stop line, published as a PointCloud2 for nav2_collision_monitor (option B of the guide).

Active only while the light is not GREEN (RED, YELLOW, or UNKNOWN) AND the car is approaching the line closer than
`active_distance`. Always publishes (an empty cloud when inactive) because the collision monitor treats a silent source
as unsafe. The stop line pose is a parameter: TODO(user) / world measurement (see traffic-light notes in README).
"""
import array
import math

import numpy as np
import rclpy
from offroad_nav2.msg import TrafficLightState
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformListener


class StopWall(Node):
    def __init__(self):
        super().__init__('stop_wall')
        d = self.declare_parameter
        self.line = d('line_xy', [0.0, 0.0]).value          # map frame
        self.yaw = d('line_yaw', 0.0).value                 # direction of travel through the line
        self.width = d('width', 10.0).value
        self.active_dist = d('active_distance', 25.0).value
        self.offset = d('wall_offset', 4.0).value   # wall this far BEFORE the line: braking distance + bumper clearance
        self.enabled = d('enabled', False).value
        self.state = TrafficLightState.UNKNOWN
        self.buf = Buffer()
        self.tl = TransformListener(self.buf, self)
        self.pub = self.create_publisher(PointCloud2, '/perception/stop_wall', 5)
        self.create_subscription(TrafficLightState, '/perception/traffic_light', lambda m: setattr(self, 'state', m.state), 5)
        self.create_timer(0.1, self.tick)

    def tick(self):
        pts = np.zeros((0, 3), np.float32)
        if self.enabled and self.state != TrafficLightState.GREEN:
            try:
                t = self.buf.lookup_transform('map', 'base_link', rclpy.time.Time()).transform.translation
                fx, fy = math.cos(self.yaw), math.sin(self.yaw)
                lx, ly = self.line[0] - self.offset * fx, self.line[1] - self.offset * fy
                along = (lx - t.x) * fx + (ly - t.y) * fy   # distance still to go to the wall
                if 0.0 < along < self.active_dist:
                    s = np.arange(-self.width / 2, self.width / 2 + 1e-6, 0.1)   # dense: the monitor needs >= min_points inside its polygon
                    pts = np.stack([lx - s * fy, ly + s * fx, np.full_like(s, 0.5)], 1).astype(np.float32)
            except Exception:
                pass
        m = PointCloud2()
        m.header.stamp, m.header.frame_id = self.get_clock().now().to_msg(), 'map'
        m.height, m.width = 1, len(pts)
        m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate('xyz')]
        m.is_bigendian, m.point_step, m.is_dense, m.row_step = False, 12, True, 12 * len(pts)
        m.data = array.array('B', pts.tobytes())
        self.pub.publish(m)


def main():
    rclpy.init()
    rclpy.spin(StopWall())


if __name__ == '__main__':
    main()
