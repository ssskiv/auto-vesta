"""perception_node: one inference engine (three separate networks, one preprocessed frame) + depth -> clouds.

Depth-provider contract: /stereo/depth (32FC1 metres, left optical frame, same stamp as the image) +
/stereo/left/camera_info. Source is pluggable (stereo_image_proc + disparity_to_depth, a RangeFinder, ...).
Every output carries the IMAGE stamp and the optical frame_id. One worker thread owns CUDA; the ROS callback only
overwrites a one-slot "latest frame" buffer.
"""
import array
import collections
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2, PointField
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from . import geometry as G
from . import postprocess as P
from .tracking import CentroidTracker, TemporalFilter, cluster_bev

RELIABLE = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
MARKING = ('geometric', 'nondrivable', 'manmade', 'vru', 'vehicle', 'natural')  # temporally filtered


def quat_to_R(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tf_to_T(t):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(t.transform.rotation)
    tr = t.transform.translation
    T[:3, 3] = [tr.x, tr.y, tr.z]
    return T


def make_cloud(pts, stamp, frame):
    m = PointCloud2()
    m.header.stamp, m.header.frame_id = stamp, frame
    m.height, m.width = 1, int(len(pts))
    m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate('xyz')]
    m.is_bigendian, m.point_step, m.is_dense = False, 12, True
    m.row_step = 12 * m.width
    m.data = array.array('B', np.ascontiguousarray(pts, np.float32).tobytes())  # fast path (bytes: ~200 ms/MB)
    return m


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        d = self.declare_parameter
        self.models_root = d('models_root', '/offroad/Models').value
        self.ckpt = {k: d(f'{k}_ckpt', f'/models/{f}').value for k, f in
                     (('seg', 'last_freespaceseg.pth'), ('contour', 'last_freespacecontour.pth'),
                      ('objseg', 'last_objectseg.pth'))}
        self.models_enabled = d('models_enabled', True).value
        self.pp = P.Params(stride=d('stride', 4).value, max_depth=d('max_depth', 20.0).value,
                           preprocess_mode=d('preprocess_mode', 'resize').value,   # TODO(user) 7.8
                           crop_top=d('crop_top', 40).value, crop_bottom=d('crop_bottom', 40).value,
                           contour_marks_nondrivable=d('contour_marks_nondrivable', True).value,
                           # footprint x -0.88..3.68, y +-0.865 (vehicle.yaml) + 0.2 m behind/side, +0.5 m ahead (hood/bumper)
                           ego_box=tuple(d('ego_box', [-1.08, 4.18, -1.065, 1.065]).value))
        self.footprint = d('footprint_frame', 'base_footprint').value
        self.odom = d('odom_frame', 'odom').value
        self.temporal = d('temporal_filter', True).value
        self.image_topic = d('image_topic', '/stereo/left/image_rect').value
        self.sim_stamp_wait = d('tf_timeout_s', 0.05).value
        self.buf = Buffer()
        self.tl = TransformListener(self.buf, self)
        self.K = None
        self.up = np.array([0.0, 0.0, 1.0])
        self.images = collections.OrderedDict()
        self.slot, self.cv = None, threading.Condition()
        self.lat = collections.deque(maxlen=200)
        self.filters = {k: TemporalFilter(0.2, 3, 2) for k in MARKING}
        self.trackers = {k: CentroidTracker() for k in ('vru', 'vehicle')}
        self.T_fp_cam = None
        self.frames = 0
        self.stage = collections.defaultdict(lambda: collections.deque(maxlen=200))
        self.wait = collections.deque(maxlen=200)
        self.plane_log = collections.deque(maxlen=200)
        self.pub = {k: self.create_publisher(PointCloud2, f'/perception/points/{k}', 2) for k in
                    ('geometric', 'nondrivable', 'clear', 'vru', 'vehicle', 'manmade', 'natural',
                     'contour_boundary', 'contour_clear_end')}
        self.pub_obj = self.create_publisher(Detection3DArray, '/perception/objects', 2)
        self.pub_lat = self.create_publisher(Float32, '/perception/latency_ms', 2)
        self.pub_vis = self.create_publisher(Image, '/perception/seg_vis', 1)
        self.create_subscription(Image, self.image_topic, self.on_image, RELIABLE)
        self.create_subscription(Image, '/stereo/depth', self.on_depth, RELIABLE)
        self.create_subscription(CameraInfo, '/stereo/left/camera_info',
                                 lambda m: setattr(self, 'K', np.array(m.k).reshape(3, 3)), RELIABLE)
        self.create_subscription(Imu, '/imu/data', self.on_imu, 5)
        self.create_timer(10.0, self.report)
        self.engine = None
        threading.Thread(target=self.worker, daemon=True).start()

    # ------------------------------------------------------------- callbacks
    def on_imu(self, m):
        self.up = quat_to_R(m.orientation).T @ np.array([0.0, 0.0, 1.0])  # world up in base_link axes

    def on_image(self, m):
        key = (m.header.stamp.sec, m.header.stamp.nanosec)
        self.images[key] = m
        while len(self.images) > 6:
            self.images.popitem(last=False)

    def on_depth(self, dm):
        key = (dm.header.stamp.sec, dm.header.stamp.nanosec)
        im = self.images.get(key)
        if im is None:
            return
        with self.cv:
            self.slot = (im, dm)       # overwrite: latest frame wins
            self.cv.notify()

    # ------------------------------------------------------------- worker
    def worker(self):
        from .engine import Engine, preprocess  # imported here: torch only in the worker
        if self.models_enabled:
            self.engine = Engine(self.models_root, self.ckpt['seg'], self.ckpt['contour'], self.ckpt['objseg'])
            self.get_logger().info(f'engine ready on {self.engine.device} (half={self.engine.half})')
        while rclpy.ok():
            with self.cv:
                while self.slot is None:
                    self.cv.wait(0.5)
                    if not rclpy.ok():
                        return
                im, dm = self.slot
                self.slot = None
            try:
                self.process(im, dm, preprocess)
            except Exception as e:  # keep the node alive
                self.get_logger().error(f'frame failed: {e!r}')

    def process(self, im, dm, preprocess):
        if self.K is None:
            return
        tk = [time.perf_counter()]
        mark = lambda n: (tk.append(time.perf_counter()), self.stage[n].append((tk[-1] - tk[-2]) * 1e3))
        self.wait.append((self.get_clock().now() - rclpy.time.Time.from_msg(dm.header.stamp)).nanoseconds * 1e-6)
        frame = im.header.frame_id
        if self.T_fp_cam is None:
            try:
                self.T_fp_cam = tf_to_T(self.buf.lookup_transform(self.footprint, frame, rclpy.time.Time()))
            except Exception:
                return
        h, w = im.height, im.width
        raw = np.frombuffer(im.data, np.uint8).reshape(h, im.step)[:, :w * (4 if 'a8' in im.encoding else 3)]
        ch = 4 if 'a8' in im.encoding else 3
        rgb = raw.reshape(h, w, ch)[..., :3]
        if im.encoding.startswith('bgr'):
            rgb = rgb[..., ::-1]
        depth = np.frombuffer(dm.data, np.float32).reshape(dm.height, dm.width)
        mark('decode')
        seg = contour = objseg = None
        if self.engine is not None:
            x = preprocess(np.ascontiguousarray(rgb), self.engine.device, self.engine.half, self.pp.preprocess_mode,
                           self.pp.crop_top, self.pp.crop_bottom)
            out = self.engine.infer_tensor(x)
            seg, contour, objseg = out.get('seg'), out.get('contour'), out.get('objseg')
        mark('nets')
        fc = P.build_clouds(seg, contour, objseg, depth, self.K, self.T_fp_cam, self.up, self.pp)
        mark('clouds')
        if fc.plane is not None:
            self.plane_log.append((fc.plane[1], float(np.degrees(np.arccos(np.clip(fc.plane[0][2], -1, 1))))))
        stamp = im.header.stamp
        # temporal filter + objects need odom<-optical at the image time
        T_odom = None
        try:
            T_odom = tf_to_T(self.buf.lookup_transform(self.odom, frame, stamp, rclpy.duration.Duration(seconds=self.sim_stamp_wait)))
        except Exception:
            pass
        clusters = {}
        for name, pts in fc.clouds.items():
            if self.temporal and name in MARKING and len(pts) and T_odom is not None:
                p_odom = pts.astype(float) @ T_odom[:3, :3].T + T_odom[:3, 3]
                keep = self.filters[name].update(p_odom[:, :2])
                if name in self.trackers:
                    clusters[name] = (p_odom[keep, :2],)
                pts = pts[keep]
            elif name in self.trackers and T_odom is not None and len(pts):
                clusters[name] = ((pts.astype(float) @ T_odom[:3, :3].T + T_odom[:3, 3])[:, :2],)
            self.pub[name].publish(make_cloud(pts, stamp, frame))
        mark('filter+publish')
        self.publish_objects(clusters, stamp)
        mark('objects')
        lat = (self.get_clock().now() - rclpy.time.Time.from_msg(stamp)).nanoseconds * 1e-6
        self.lat.append(lat)
        self.pub_lat.publish(Float32(data=float(lat)))
        self.frames += 1
        if self.frames % 6 == 0 and seg is not None and self.pub_vis.get_subscription_count():
            self.publish_vis(rgb, seg, objseg, stamp)

    def publish_objects(self, clusters, stamp):
        arr = Detection3DArray()
        arr.header.stamp, arr.header.frame_id = stamp, self.odom
        for name, (xy,) in clusters.items():
            for tid, c in self.trackers[name].update(cluster_bev(xy)):
                d = Detection3D()
                d.header = arr.header
                d.id = f'{name}-{tid}'
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id, hyp.hypothesis.score = name, 1.0
                d.results.append(hyp)
                d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z = float(c['center'][0]), float(c['center'][1]), 0.0
                d.bbox.size.x, d.bbox.size.y, d.bbox.size.z = float(c['size'][0]), float(c['size'][1]), 1.0
                arr.detections.append(d)
        self.pub_obj.publish(arr)

    def publish_vis(self, rgb, seg, objseg, stamp):
        import cv2
        small = cv2.resize(np.ascontiguousarray(rgb), (640, 320))
        pal = np.array([[0, 0, 0], [0, 200, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], np.uint8)
        vis = small.copy()
        vis[seg == 1] = (0.55 * vis[seg == 1] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
        cols = {1: (255, 255, 0), 2: (255, 0, 0), 3: (0, 0, 255), 4: (255, 0, 255)}
        if objseg is not None:
            for c, col in cols.items():
                vis[objseg == c] = (0.5 * vis[objseg == c] + 0.5 * np.array(col)).astype(np.uint8)
        m = Image()
        m.header.stamp, m.header.frame_id = stamp, 'seg_vis'
        m.height, m.width, m.encoding, m.step = 320, 640, 'rgb8', 640 * 3
        m.data = array.array('B', vis.tobytes())
        self.pub_vis.publish(m)

    def report(self):
        if self.lat:
            v = np.array(self.lat)
            self.get_logger().info(f'image->cloud latency p50 {np.percentile(v, 50):.0f} ms p95 {np.percentile(v, 95):.0f} ms '
                                   f'({len(v)} frames, {self.frames} total); depth-arrival->now p50 {np.percentile(self.wait, 50):.0f} ms; '
                                   + (f'ground plane in {self.footprint}: d median {np.median([a for a, _ in self.plane_log]):+.3f} m, tilt {np.median([b for _, b in self.plane_log]):.2f} deg; ' if self.plane_log else '')
                                   + ' '.join(f'{k} {np.percentile(x, 50):.0f}' for k, x in self.stage.items()) + ' (stage p50 ms)')


def main():
    rclpy.init()
    n = PerceptionNode()
    from rclpy.executors import MultiThreadedExecutor
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(n)
    ex.spin()


if __name__ == '__main__':
    main()
