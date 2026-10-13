"""Gym env over ROS 2 + Webots, for fine-tuning and validating the policy.

Needs robot.launch.py (vesta_driver: /odom, /cmd_vel, /vesta/teleport) and
nav2.launch.py (local_costmap inside controller_server, map->odom TF) running.
The env drives /cmd_vel itself; controller_server just idles.
"""
import math
import time

import gymnasium as gym
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.msg import Costmap
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from tf2_ros import Buffer, TransformListener
from gymnasium import spaces

from . import arbiter
from . import config as C
from .obs import CostmapView, PathTracker, build_patch, build_vec, footprint_world
from .paths import load_paths
from .reward import Episode, override_penalty, speed_limit, step_reward
from .sim2d import observation_space


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class WebotsEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, paths_npz, max_speed=C.MAX_SPEED, path_frame='map', seed=None):
        self.paths = load_paths(paths_npz)
        self.max_speed = max_speed
        self.path_frame = path_frame
        self.observation_space = observation_space()
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.rng = np.random.default_rng(seed)

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node('vesta_rl_env', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE)
        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', qos)
        self.teleport_pub = self.node.create_publisher(PoseStamped, '/vesta/teleport', 1)
        self.node.create_subscription(Odometry, '/odom', self._on_odom, qos_profile_sensor_data)
        self.node.create_subscription(Costmap, '/local_costmap/costmap_raw', self._on_costmap, 1)
        self.node.create_subscription(Clock, '/clock', self._on_clock, 10)
        self.clear_client = self.node.create_client(ClearEntireCostmap, '/local_costmap/clear_entirely_local_costmap')
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self.node)
        self.odom = None
        self.costmap = None
        self.costmap_stamp = 0.0
        self.now = 0.0

    # ---- ROS plumbing --------------------------------------------------
    def _on_odom(self, msg):
        self.odom = msg

    def _on_costmap(self, msg):
        m = msg.metadata
        data = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(m.size_y, m.size_x)
        self.costmap = CostmapView(data, m.origin.position.x, m.origin.position.y, m.resolution)
        self.costmap_stamp = self.now

    def _on_clock(self, msg):
        self.now = msg.clock.sec + msg.clock.nanosec * 1e-9

    def _wait_sim(self, seconds, timeout=30.0):
        target = self.now + seconds
        deadline = time.monotonic() + timeout
        while self.now < target:
            rclpy.spin_once(self.node, timeout_sec=0.005)
            if time.monotonic() > deadline:
                raise RuntimeError('no /clock from Webots')

    def _wait_for(self, cond, timeout=30.0):
        deadline = time.monotonic() + timeout
        while not cond():
            rclpy.spin_once(self.node, timeout_sec=0.01)
            if time.monotonic() > deadline:
                raise RuntimeError('timed out waiting for /odom or /local_costmap/costmap_raw')

    def _publish(self, v, delta):
        lin, ang = C.command_to_twist(v, delta)
        msg = Twist()
        msg.linear.x, msg.angular.z = lin, ang
        self.cmd_pub.publish(msg)

    def _pose(self):
        p = self.odom.pose.pose
        return p.position.x, p.position.y, _yaw(p.orientation)

    def _path_to_odom(self, xy):
        t = self.tf.lookup_transform('odom', self.path_frame, rclpy.time.Time(),
                                     timeout=rclpy.duration.Duration(seconds=5.0))
        tx, ty = t.transform.translation.x, t.transform.translation.y
        yaw = _yaw(t.transform.rotation)
        c, s = math.cos(yaw), math.sin(yaw)
        return np.stack([tx + c * xy[:, 0] - s * xy[:, 1], ty + s * xy[:, 0] + c * xy[:, 1]], axis=1)

    # ---- gym API -------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._publish(0.0, 0.0)
        self._wait_for(lambda: self.odom is not None and self.costmap is not None)

        path = self._path_to_odom(self.paths[self.rng.integers(len(self.paths))])
        self.tracker = PathTracker(path)
        x0, y0 = path[0]
        x1, y1 = path[min(2, len(path) - 1)]
        yaw = math.atan2(y1 - y0, x1 - x0)
        for _ in range(5):
            msg = PoseStamped()
            msg.header.frame_id = 'odom'
            msg.pose.position.x, msg.pose.position.y = float(x0), float(y0)
            msg.pose.orientation.z, msg.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            self.teleport_pub.publish(msg)
            self._wait_sim(1.0)
            px, py, _ = self._pose()
            if math.hypot(px - x0, py - y0) < 1.0:
                break
        else:
            raise RuntimeError('teleport failed: is vesta_driver running with /vesta/teleport?')

        if self.clear_client.wait_for_service(timeout_sec=2.0):
            rclpy.spin_until_future_complete(self.node, self.clear_client.call_async(ClearEntireCostmap.Request()),
                                             timeout_sec=2.0)
        stamp = self.now
        self._wait_for(lambda: self.costmap_stamp > stamp + 0.3)

        self.last_v, self.last_delta = 0.0, 0.0
        max_steps = int((self.tracker.length / 1.5 + 30.0) / C.DT)
        self.episode = Episode(self.tracker.length, max_steps)
        obs, (s_proj, _, _) = self._obs()
        self.prev_s = s_proj
        return obs, {}

    def _obs(self):
        px, py, yaw = self._pose()
        tw = self.odom.twist.twist
        vec, proj = build_vec(self.tracker, px, py, yaw, tw.linear.x, tw.angular.z,
                              self.last_v, self.last_delta, self.max_speed)
        patch = build_patch(self.costmap, px, py, yaw)
        return {'costmap': patch[None], 'vec': vec}, proj

    def step(self, action):
        v_lim = speed_limit(self.tracker, self.prev_s, self.max_speed)
        v_cmd, d_cmd = C.action_to_command(float(action[0]), float(action[1]), v_lim)
        v_cmd, d_cmd = C.rate_limit(v_cmd, d_cmd, self.last_v, self.last_delta)
        px, py, yaw = self._pose()
        proposed = (v_cmd, d_cmd)
        (v_cmd, d_cmd), source = arbiter.arbitrate(  # same safety layer as the Nav2 plugin
            self.costmap, self.tracker, px, py, yaw, self.odom.twist.twist.linear.x,
            self.last_v, self.last_delta, (v_cmd, d_cmd), v_lim)
        dv, dd = v_cmd - self.last_v, d_cmd - self.last_delta
        self.last_v, self.last_delta = v_cmd, d_cmd
        self._publish(v_cmd, d_cmd)
        self._wait_sim(C.DT)

        obs, (s_proj, cte, heading_err) = self._obs()
        px, py, yaw = self._pose()
        fx, fy = footprint_world(px, py, yaw)
        cost = self.costmap.cost_at(fx, fy)
        collided = bool((cost == C.LETHAL).any())
        cost = cost.astype(np.float32)
        cost[cost == C.NO_INFORMATION] = 0.0
        overspeed = max(0.0, self.odom.twist.twist.linear.x - speed_limit(self.tracker, s_proj, self.max_speed))
        reward = step_reward(s_proj - self.prev_s, cte, heading_err, float(cost.mean() / C.LETHAL),
                             dv, dd, self.max_speed, overspeed)
        if source != arbiter.NN:
            reward -= override_penalty(proposed, (v_cmd, d_cmd), self.max_speed)
        self.prev_s = s_proj
        ex, ey = self.tracker.xy[-1]
        r_end, terminated, truncated, success = self.episode.check(s_proj, math.hypot(px - ex, py - ey), cte, collided)
        if terminated or truncated:
            self._publish(0.0, d_cmd)
        info = {'source': source}
        if terminated or truncated:
            info.update(is_success=success, reason=self.episode.reason)
        return obs, reward + r_end, terminated, truncated, info

    def close(self):
        self._publish(0.0, 0.0)
        self.node.destroy_node()
