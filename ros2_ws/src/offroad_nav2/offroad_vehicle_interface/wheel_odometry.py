"""/joint_states (rear wheel angles) -> /wheel/odometry (twist.linear.x only), for the local EKF."""
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .logic import WheelSpeed

REAR = ('rear_left_wheel_joint', 'rear_right_wheel_joint')


class WheelOdometry(Node):
    def __init__(self):
        super().__init__('wheel_odometry')
        self.ws = WheelSpeed(self.declare_parameter('wheel_radius', 0.315).value,
                             self.declare_parameter('sign', 1.0).value)  # TODO: check sign convention per sim/car
        self.sigma = self.declare_parameter('sigma_v', 0.1).value
        self.pub = self.create_publisher(Odometry, '/wheel/odometry', 10)
        self.create_subscription(JointState, '/joint_states', self.cb, 10)

    def cb(self, m):
        try:
            a = [m.position[m.name.index(n)] for n in REAR]
        except ValueError:
            return
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        v = self.ws.update(a, t)
        if v is None:
            return
        o = Odometry()
        o.header.stamp, o.header.frame_id, o.child_frame_id = m.header.stamp, 'odom', 'base_link'
        o.twist.twist.linear.x = float(v)
        o.twist.covariance[0] = self.sigma ** 2
        for i in (7, 14, 21, 28, 35):
            o.twist.covariance[i] = 1e3
        o.pose.covariance[0] = 1e3
        self.pub.publish(o)


def main():
    rclpy.init()
    rclpy.spin(WheelOdometry())


if __name__ == '__main__':
    main()
