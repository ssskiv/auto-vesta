"""Webots-only adapter so the real-car localization stack (navsat_transform + EKFs) runs unchanged.

 * /gps/fix (Webots: PointStamped, world metres of the antenna) -> /gps/fix_ll (NavSatFix). The antenna is moved to
   base_link with the IMU attitude and the gps_link TF (lever arm 0.3 m fwd, 1.0 m up). The datum is the map origin,
   i.e. world (2.2, -1.6) (world = map + (2.2, -1.6), maps/robocross.yaml). Latitude/longitude of the datum is a
   placeholder: TODO(user) real RoboCross venue.
 * /imu/data (the driver's sim-stamped republish of /vesta/imu/data, which has WALL stamps; covariances are zero) -> /imu/data_cov with covariances filled in.
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu, NavSatFix
from tf2_ros import Buffer, TransformListener

R_EARTH = 6378137.0


def quat_to_R(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class SimSensorAdapter(Node):
    def __init__(self):
        super().__init__('sim_sensor_adapter')
        self.lat0 = self.declare_parameter('datum_lat', 54.3282).value   # TODO(user)
        self.lon0 = self.declare_parameter('datum_lon', 48.3866).value   # TODO(user)
        self.off = np.array(self.declare_parameter('map_origin_in_world', [2.2, -1.6]).value, float)
        self.sigma_xy = self.declare_parameter('gps_sigma_xy', 0.3).value
        self.q = None
        self.lever = None
        self.buf = Buffer()
        self.tl = TransformListener(self.buf, self)
        qr = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_ll = self.create_publisher(NavSatFix, '/gps/fix_ll', qr)
        self.pub_imu = self.create_publisher(Imu, '/imu/data_cov', qr)
        self.create_subscription(Imu, '/imu/data', self.on_imu, qos_profile_sensor_data)
        self.create_subscription(PointStamped, '/gps/fix', self.on_gps, qos_profile_sensor_data)

    def on_imu(self, m):
        self.q = m.orientation
        for i in (0, 4, 8):
            m.orientation_covariance[i] = 1e-4
            m.angular_velocity_covariance[i] = 1e-4
            m.linear_acceleration_covariance[i] = 1e-2
        self.pub_imu.publish(m)

    def on_gps(self, m):
        if self.q is None:
            return
        if self.lever is None:
            try:
                t = self.buf.lookup_transform('base_link', m.header.frame_id or 'gps_link', rclpy.time.Time()).transform.translation
                self.lever = np.array([t.x, t.y, t.z])
            except Exception:
                return
        p = np.array([m.point.x, m.point.y, m.point.z]) - quat_to_R(self.q) @ self.lever  # base_link in world
        east, north = p[0] - self.off[0], p[1] - self.off[1]                              # map metres
        f = NavSatFix()
        f.header.stamp, f.header.frame_id = self.get_clock().now().to_msg(), 'base_link'
        f.status.status, f.status.service = 0, 1
        f.latitude = self.lat0 + math.degrees(north / R_EARTH)
        f.longitude = self.lon0 + math.degrees(east / (R_EARTH * math.cos(math.radians(self.lat0))))
        f.altitude = float(p[2])
        f.position_covariance = [self.sigma_xy ** 2, 0.0, 0.0, 0.0, self.sigma_xy ** 2, 0.0, 0.0, 0.0, 1.0]
        f.position_covariance_type = 2
        self.pub_ll.publish(f)


def main():
    rclpy.init()
    rclpy.spin(SimSensorAdapter())


if __name__ == '__main__':
    main()
