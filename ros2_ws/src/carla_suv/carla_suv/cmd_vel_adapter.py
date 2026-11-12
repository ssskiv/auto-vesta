"""/cmd_vel (Twist) -> CarlaEgoVehicleControl, same Ackermann conversion as vesta_driver.py."""
import math
import time

import rclpy
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus
from geometry_msgs.msg import Twist
from rclpy.node import Node

WHEELBASE = 2.648       # m, LadaVesta.proto
MAX_STEER = 0.6         # rad, vesta_driver.py MAX_STEER
COMMAND_TIMEOUT = 0.5   # s, no /cmd_vel for this long -> full brake


def clamp(value, low, high):
    return max(low, min(high, value))


def steer_angle(linear_x, angular_z):
    """Left-positive front wheel angle (rad) for a Twist, clamped like the Webots driver."""
    return clamp(math.atan2(angular_z * WHEELBASE, linear_x), -MAX_STEER, MAX_STEER)


class SpeedController:
    """PI throttle with a brake for overspeed. Output (throttle, brake, reverse)."""

    def __init__(self, kp=0.6, ki=0.4, kb=0.5, max_integral=2.0):
        self.kp, self.ki, self.kb, self.max_integral = kp, ki, kb, max_integral
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def update(self, target, speed, dt):
        """target: signed m/s from cmd_vel; speed: signed m/s from the vehicle."""
        reverse = target < 0.0
        error = abs(target) - (-speed if reverse else speed)
        if abs(target) < 1e-3:
            self.reset()
            return 0.0, 1.0 if abs(speed) > 0.1 else 0.3, reverse
        self.integral = clamp(self.integral + error * dt, -self.max_integral, self.max_integral)
        out = self.kp * error + self.ki * self.integral
        if out >= 0.0:
            return clamp(out, 0.0, 1.0), 0.0, reverse
        return 0.0, clamp(-out * self.kb, 0.0, 1.0), reverse


class CmdVelAdapter(Node):
    def __init__(self):
        super().__init__('cmd_vel_adapter')
        role = self.declare_parameter('role_name', 'vesta').value
        # Physical steering range the CARLA vehicle maps steer=+-1 to (rad). mkz_2017: ~70 deg.
        self.max_wheel_angle = self.declare_parameter('carla_max_steer_angle', 1.22).value
        rate = self.declare_parameter('rate', 50.0).value

        self.target = 0.0
        self.angle = 0.0
        self.speed = 0.0
        self.last_cmd = time.monotonic()
        self.last_tick = time.monotonic()
        self.pid = SpeedController()

        self.create_subscription(Twist, '/cmd_vel', self.on_cmd_vel, 10)
        self.create_subscription(
            CarlaEgoVehicleStatus, f'/carla/{role}/vehicle_status', self.on_status, 10)
        self.pub = self.create_publisher(
            CarlaEgoVehicleControl, f'/carla/{role}/vehicle_control_cmd', 10)
        self.create_timer(1.0 / rate, self.tick)

    def on_cmd_vel(self, msg):
        self.target = msg.linear.x
        if abs(msg.linear.x) > 1e-3:   # keep the last angle at standstill, like the Webots driver
            self.angle = steer_angle(msg.linear.x, msg.angular.z)
        self.last_cmd = time.monotonic()

    def on_status(self, msg):
        # CarlaEgoVehicleStatus.velocity is the signed forward speed in m/s.
        self.speed = msg.velocity

    def tick(self):
        now = time.monotonic()
        dt = now - self.last_tick
        self.last_tick = now

        cmd = CarlaEgoVehicleControl()
        cmd.header.stamp = self.get_clock().now().to_msg()
        # CARLA steer > 0 is right, cmd_vel angular.z > 0 is left.
        cmd.steer = float(clamp(-self.angle / self.max_wheel_angle, -1.0, 1.0))
        if now - self.last_cmd > COMMAND_TIMEOUT:
            self.pid.reset()
            cmd.throttle, cmd.brake, cmd.reverse = 0.0, 1.0, False
        else:
            throttle, brake, reverse = self.pid.update(self.target, self.speed, dt)
            cmd.throttle, cmd.brake, cmd.reverse = float(throttle), float(brake), bool(reverse)
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
