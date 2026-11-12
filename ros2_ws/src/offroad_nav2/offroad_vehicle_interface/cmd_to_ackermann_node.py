"""Real-car path: Twist -> steering angle + speed -> DbwAdapter (logging stub until decision 7.6). Not used in sim."""
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool

from .dbw_adapter_base import LoggingDbwAdapter
from .logic import ZeroClampWatchdog, steer_from_twist


class CmdToAckermann(Node):
    def __init__(self):
        super().__init__('cmd_to_ackermann')
        self.L = self.declare_parameter('wheelbase', 2.648).value
        self.max_steer = self.declare_parameter('max_steer', 0.6).value
        self.max_rate = self.declare_parameter('max_steer_rate', 0.5).value  # rad/s, TODO(user): measure
        self.adapter = LoggingDbwAdapter()  # TODO(user) decision 7.6: replace with the OSCC/ros2cco adapter
        self.engaged = False
        self.wd = ZeroClampWatchdog(0.3)
        self.steer = 0.0
        self.t_prev = None
        self.pub = self.create_publisher(AckermannDriveStamped, '/vehicle/ackermann_cmd', 10)
        self.create_subscription(Twist, '/cmd_vel_nav', lambda m: self.wd.update(m.linear.x, m.angular.z, self.now()), 10)
        self.create_subscription(Bool, '/vehicle/engage', lambda m: setattr(self, 'engaged', m.data), 10)
        self.create_subscription(Bool, '/vehicle/emergency_stop', lambda m: self.adapter.emergency_stop() if m.data else None, 10)
        self.create_timer(0.05, self.tick)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def tick(self):
        t = self.now()
        dt = 0.05 if self.t_prev is None else t - self.t_prev
        self.t_prev = t
        v, w, stale = self.wd.output(t)
        if not self.engaged or self.adapter.estop:
            v = 0.0
        self.steer = steer_from_twist(v, w, self.L, self.steer, self.max_steer, max_rate=self.max_rate, dt=dt)
        self.adapter.send(self.steer, v, 0.0)
        m = AckermannDriveStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.drive.steering_angle, m.drive.speed = float(self.steer), float(v)
        self.pub.publish(m)


def main():
    rclpy.init()
    rclpy.spin(CmdToAckermann())


if __name__ == '__main__':
    main()
