"""/cmd_vel_nav (Twist) -> /cmd_vel with linear.x >= 0 and a 0.3 s watchdog. Mandatory in sim: the sim adapter
reverses on negative speed."""
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from .logic import ZeroClampWatchdog


class ZeroClampRelay(Node):
    def __init__(self):
        super().__init__('zero_clamp_relay')
        self.logic = ZeroClampWatchdog(self.declare_parameter('watchdog_s', 0.3).value)
        self.pub = self.create_publisher(Twist, self.declare_parameter('out_topic', '/cmd_vel').value, 10)
        self.create_subscription(Twist, self.declare_parameter('in_topic', '/cmd_vel_nav').value, self.cb, 10)
        self.create_timer(0.05, self.tick)
        self.last_log = 0

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def cb(self, m):
        n0 = self.logic.clamped
        self.logic.update(m.linear.x, m.angular.z, self.now())
        if self.logic.clamped != n0 and self.now() - self.last_log > 2.0:
            self.last_log = self.now()
            self.get_logger().warn(f'clamped negative linear.x={m.linear.x:.2f} to 0 ({self.logic.clamped} total)')
        self.emit(self.now())

    def tick(self):
        v, w, stale = self.logic.output(self.now())
        if stale:
            self.emit(self.now())

    def emit(self, t):
        v, w, _ = self.logic.output(t)
        m = Twist()
        m.linear.x, m.angular.z = float(v), float(w)
        self.pub.publish(m)


def main():
    rclpy.init()
    rclpy.spin(ZeroClampRelay())


if __name__ == '__main__':
    main()
