"""Webots-only: re-stamp camera images with SIM time and make them stereo-ready.

Webots /camera/* use wall-clock stamps (README, "Известные ограничения"), so TF lookups at the image stamp fail.
This node subscribes to the raw Webots stereo topics, stamps each left/right pair with the sim time at which the
first image of the pair arrived (/clock, +-1 sim step), and republishes
  /stereo/{left,right}/image_rect   (same pixels; Webots cameras have no distortion, so raw == rectified)
  /stereo/{left,right}/camera_info  (K from Webots, R = I, D = 0, P with Tx = -fx*baseline on the right camera)
  /stereo_crop/{left,right}/image_rect + camera_info: rows [crop_top, crop_bottom) only (the matcher does not need
      sky or hood; ~2.5x faster SGBM). cy is shifted, camera_info.roi holds the offset and the full size.
Left and right of a pair get the identical stamp (exact sync for stereo_image_proc).
"""
import array

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class StampRelay(Node):
    def __init__(self):
        super().__init__('stamp_relay')
        self.baseline = self.declare_parameter('baseline', 0.20).value
        self.in_ns = self.declare_parameter('in_ns', '/stereo_raw').value
        self.suffix = self.declare_parameter('in_image_suffix', 'image_color').value
        self.mono = self.declare_parameter('mono_topic', '/camera/image_raw/image_color').value
        self.crop = (int(self.declare_parameter('crop_top', 340).value), int(self.declare_parameter('crop_bottom', 624).value))
        self.pair_tol_ns = int(self.declare_parameter('pair_tolerance_ms', 8.0).value * 1e6)
        self.min_period = 1.0 / self.declare_parameter('max_rate_hz', 10.0).value  # SGBM would otherwise queue (latency)
        self.last_pub = -1e9
        self.skip_keys = {}
        self.pairs = {}  # original wall stamp (ns) -> sim stamp (Time msg)
        self.K = {}
        out = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        # best-effort drops fragments of the 3.7 MB images over UDP; Webots publishes reliable
        inq = QoSProfile(depth=3, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = {}
        for side in ('left', 'right'):
            self.pub[side] = (self.create_publisher(Image, f'/stereo/{side}/image_rect', out),
                              self.create_publisher(CameraInfo, f'/stereo/{side}/camera_info', out))
            self.create_subscription(Image, f'{self.in_ns}/{side}/{self.suffix}',
                                     lambda m, s=side: self.on_image(m, s), inq)
            self.create_subscription(CameraInfo, f'{self.in_ns}/{side}/camera_info',
                                     lambda m, s=side: self.K.__setitem__(s, np.array(m.k)), qos_profile_sensor_data)
        self.cpub = {side: (self.create_publisher(Image, f'/stereo_crop/{side}/image_rect', out),
                            self.create_publisher(CameraInfo, f'/stereo_crop/{side}/camera_info', out))
                     for side in ('left', 'right')}
        self.mono_pub = self.create_publisher(Image, '/camera/image_rect', out)
        self.create_subscription(Image, self.mono, self.on_mono, inq)
        self.get_logger().info(f'restamping with sim time (use_sim_time={self.get_parameter("use_sim_time").value})')

    def sim_stamp(self, orig):
        key = orig.sec * 10**9 + orig.nanosec
        # left/right of one Webots step carry wall stamps a few ms apart: treat them as one pair
        for k in self.pairs:
            if abs(k - key) < self.pair_tol_ns:
                return self.pairs[k]
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9
        self.pairs[key] = now.to_msg()
        self.skip_keys[key] = (t - self.last_pub) < self.min_period * 0.9
        if not self.skip_keys[key]:
            self.last_pub = t
        if len(self.pairs) > 32:
            for k in sorted(self.pairs)[:-16]:
                del self.pairs[k]
                self.skip_keys.pop(k, None)
        return self.pairs[key]

    def skipped(self, orig):
        key = orig.sec * 10**9 + orig.nanosec
        for k in self.skip_keys:
            if abs(k - key) < self.pair_tol_ns:
                return self.skip_keys[k]
        return False

    def on_mono(self, msg):
        msg.header.stamp = self.get_clock().now().to_msg()
        self.mono_pub.publish(msg)

    def on_image(self, msg, side):
        K = self.K.get(side)
        if K is None:
            return  # wait for the first camera_info
        stamp = self.sim_stamp(msg.header.stamp)
        if self.skipped(msg.header.stamp):
            return
        frame = msg.header.frame_id
        msg.header.stamp = stamp
        info = CameraInfo()
        info.header.stamp, info.header.frame_id = stamp, frame
        info.width, info.height = msg.width, msg.height
        info.distortion_model = 'plumb_bob'
        info.d = [0.0] * 5
        info.k = [float(v) for v in K]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        tx = -K[0] * self.baseline if side == 'right' else 0.0
        info.p = [float(v) for v in (K[0], 0.0, K[2], tx, 0.0, K[4], K[5], 0.0, 0.0, 0.0, 1.0, 0.0)]
        if side == 'left':
            self.pub[side][0].publish(msg)
            self.pub[side][1].publish(info)
        # cropped copy for the stereo matcher
        r0, r1 = self.crop
        h, step = msg.height, msg.step
        rows = np.frombuffer(msg.data, np.uint8).reshape(h, step)[r0:r1]
        cm = Image()
        cm.header, cm.height, cm.width, cm.encoding, cm.is_bigendian, cm.step = msg.header, r1 - r0, msg.width, msg.encoding, msg.is_bigendian, step
        cm.data = array.array('B', rows.tobytes())  # array.array: bytes assignment costs ~200 ms/MB in rclpy Humble
        ci = CameraInfo()
        ci.header, ci.width, ci.height = info.header, info.width, r1 - r0
        ci.distortion_model, ci.d, ci.k, ci.r = info.distortion_model, info.d, list(info.k), info.r
        ci.k[5] = float(K[5] - r0)
        ci.p = list(info.p)
        ci.p[6] = float(info.p[6] - r0)
        ci.roi.y_offset, ci.roi.height, ci.roi.width = r0, r1 - r0, info.width
        self.cpub[side][0].publish(cm)
        self.cpub[side][1].publish(ci)


def main():
    rclpy.init()
    rclpy.spin(StampRelay())


if __name__ == '__main__':
    main()
