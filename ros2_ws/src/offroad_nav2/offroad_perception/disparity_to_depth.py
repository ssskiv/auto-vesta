"""stereo_msgs/DisparityImage -> /stereo/depth (32FC1, metres, left-camera optical frame, NaN = invalid).

This is the depth-provider contract of perception_node: /stereo/depth + /stereo/left/camera_info.
Any other source (Webots RangeFinder, lidar projection, real stereo) just has to publish the same two topics."""
import array

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage


class DisparityToDepth(Node):
    def __init__(self):
        super().__init__('disparity_to_depth')
        self.min_depth = self.declare_parameter('min_depth', 1.0).value
        self.max_depth = self.declare_parameter('max_depth', 60.0).value
        self.crop_top = int(self.declare_parameter('crop_top', 340).value)
        self.full_h = int(self.declare_parameter('full_height', 720).value)
        q = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(Image, '/stereo/depth', q)
        self.create_subscription(DisparityImage, '/stereo/disparity', self.cb, q)

    def cb(self, msg):
        im = msg.image
        d = np.frombuffer(im.data, dtype=np.float32).reshape(im.height, im.step // 4)[:, :im.width]
        with np.errstate(divide='ignore', invalid='ignore'):
            z = msg.f * msg.t / d
        z[~np.isfinite(z) | (d <= max(msg.min_disparity, 0.0)) | (z < self.min_depth) | (z > self.max_depth)] = np.nan
        if self.full_h > im.height:  # matcher saw rows [crop_top, crop_top+h): pad back to the full frame
            full = np.full((self.full_h, im.width), np.nan, np.float32)
            full[self.crop_top:self.crop_top + im.height] = z
            z = full
        out = Image()
        out.header = msg.header
        out.height, out.width, out.encoding = z.shape[0], z.shape[1], '32FC1'
        out.is_bigendian, out.step = 0, z.shape[1] * 4
        # array.array: assigning bytes/ndarray to a uint8[] field runs a ~200 ms per-element check in rclpy (Humble)
        out.data = array.array('B', np.ascontiguousarray(z, dtype=np.float32).tobytes())
        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(DisparityToDepth())


if __name__ == '__main__':
    main()
