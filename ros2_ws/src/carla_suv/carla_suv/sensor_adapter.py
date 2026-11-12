"""carla-ros-bridge topics -> the topics/frames webots_ros2_suv publishes.

  /carla/<role>/imu                 -> /imu/data           (imu_link)
  /carla/<role>/lidar               -> /lidar              (lidar_link)
  /carla/<role>/gnss                -> /gps/fix            (gps_link)
  /carla/<role>/rgb_camera/image    -> /camera/image_raw   (camera_link)
  /carla/<role>/radar               -> /radar              (radar_link, xyz32, z = 0)
  /carla/<role>/stereo_left|right/image        -> /stereo/left|right/image_raw    (stereo_left|right_link)
  /carla/<role>/stereo_left|right/camera_info  -> /stereo/left|right/camera_info
  /carla/<role>/odometry            -> TF odom -> base_link
  /carla/<role>/vehicle_status      -> /joint_states
"""
import math

import rclpy
from carla_msgs.msg import CarlaEgoVehicleStatus
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Imu, JointState, NavSatFix, PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import TransformBroadcaster

TIRE_RADIUS = 0.315
MAX_STEER = 0.6
STEER_JOINTS = ('front_left_steer_joint', 'front_right_steer_joint')
WHEEL_JOINTS = ('front_left_wheel_joint', 'front_right_wheel_joint',
                'rear_left_wheel_joint', 'rear_right_wheel_joint')


def rotate_x(q, dx):
    """World-frame vector (dx, 0, 0) rotated by quaternion q (x, y, z, w), as (x, y, z)."""
    x, y, z, w = q
    return (dx * (1 - 2 * (y * y + z * z)), dx * 2 * (x * y + w * z), dx * 2 * (x * z - w * y))


class SensorAdapter(Node):
    def __init__(self):
        super().__init__('sensor_adapter')
        role = self.declare_parameter('role_name', 'vesta').value
        # Distance from CARLA's actor origin to the rear axle (base_link), along +x.
        self.rear_axle_offset = self.declare_parameter('rear_axle_offset', 1.4).value
        self.max_wheel_angle = self.declare_parameter('carla_max_steer_angle', 1.22).value

        self.origin = None
        self.wheel_angle = 0.0
        self.last_status_stamp = None
        self.tf = TransformBroadcaster(self)

        out_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)
        in_qos = qos_profile_sensor_data
        base = f'/carla/{role}'

        self.imu_pub = self.create_publisher(Imu, '/imu/data', out_qos)
        self.lidar_pub = self.create_publisher(PointCloud2, '/lidar', out_qos)
        self.radar_pub = self.create_publisher(PointCloud2, '/radar', out_qos)
        self.gps_pub = self.create_publisher(NavSatFix, '/gps/fix', out_qos)
        self.cam_pub = self.create_publisher(Image, '/camera/image_raw', out_qos)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', out_qos)

        self.create_subscription(Imu, f'{base}/imu', lambda m: self.relay(m, self.imu_pub, 'imu_link'), in_qos)
        self.create_subscription(PointCloud2, f'{base}/lidar', lambda m: self.relay(m, self.lidar_pub, 'lidar_link'), in_qos)
        self.create_subscription(NavSatFix, f'{base}/gnss', lambda m: self.relay(m, self.gps_pub, 'gps_link'), in_qos)
        self.create_subscription(Image, f'{base}/rgb_camera/image', lambda m: self.relay(m, self.cam_pub, 'camera_link'), in_qos)
        for side in ('left', 'right'):   # present only with sensors:=stereo
            frame = f'stereo_{side}_link'
            image_pub = self.create_publisher(Image, f'/stereo/{side}/image_raw', out_qos)
            info_pub = self.create_publisher(CameraInfo, f'/stereo/{side}/camera_info', out_qos)
            self.create_subscription(Image, f'{base}/stereo_{side}/image',
                                     lambda m, p=image_pub, f=frame: self.relay(m, p, f), in_qos)
            self.create_subscription(CameraInfo, f'{base}/stereo_{side}/camera_info',
                                     lambda m, p=info_pub, f=frame: self.relay(m, p, f), in_qos)
        self.create_subscription(PointCloud2, f'{base}/radar', self.on_radar, in_qos)
        self.create_subscription(Odometry, f'{base}/odometry', self.on_odometry, in_qos)
        self.create_subscription(CarlaEgoVehicleStatus, f'{base}/vehicle_status', self.on_status, in_qos)

    @staticmethod
    def relay(msg, publisher, frame_id):
        msg.header.frame_id = frame_id
        publisher.publish(msg)

    def on_radar(self, msg):
        # The bridge puts the detections in x, y, z (ROS axes); Webots' /radar is flat (z = 0).
        pts = point_cloud2.read_points(msg, field_names=('x', 'y'), skip_nans=True)
        cloud = point_cloud2.create_cloud_xyz32(
            msg.header, [(float(p[0]), float(p[1]), 0.0) for p in pts])
        cloud.header.frame_id = 'radar_link'
        self.radar_pub.publish(cloud)

    def on_odometry(self, msg):
        """odom -> base_link: translation relative to the first sample, absolute orientation
        (the same convention as vesta_driver.py). base_link is the rear axle, not the actor origin."""
        o = msg.pose.pose.orientation
        q = (o.x, o.y, o.z, o.w)
        p = msg.pose.pose.position
        axle = rotate_x(q, -self.rear_axle_offset)
        pos = (p.x + axle[0], p.y + axle[1], p.z + axle[2])
        if self.origin is None:
            self.origin = pos
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = pos[0] - self.origin[0]
        t.transform.translation.y = pos[1] - self.origin[1]
        t.transform.translation.z = pos[2] - self.origin[2]
        t.transform.rotation = o
        self.tf.sendTransform(t)

    def on_status(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = 0.0 if self.last_status_stamp is None else max(0.0, stamp - self.last_status_stamp)
        self.last_status_stamp = stamp
        wheel_rate = msg.velocity / TIRE_RADIUS
        # CARLA steer > 0 is right; joint angles are left-positive.
        steer = max(-MAX_STEER, min(MAX_STEER, -msg.control.steer * self.max_wheel_angle))
        self.wheel_angle = (self.wheel_angle + wheel_rate * dt) % (2 * math.pi)

        js = JointState()
        js.header.stamp = msg.header.stamp
        js.name = list(STEER_JOINTS) + list(WHEEL_JOINTS)
        js.position = [steer] * len(STEER_JOINTS) + [self.wheel_angle] * len(WHEEL_JOINTS)
        js.velocity = [0.0] * len(STEER_JOINTS) + [wheel_rate] * len(WHEEL_JOINTS)
        self.joint_pub.publish(js)


def main(args=None):
    rclpy.init(args=args)
    node = SensorAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
