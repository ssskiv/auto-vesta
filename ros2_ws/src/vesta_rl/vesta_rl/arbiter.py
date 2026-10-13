"""The plugin's arbiter (vesta_neural_controller NeuralController::computeVelocityCommands)
in Python, so training sees the same safety layer as Nav2.

Each tick the policy's command and a pure-pursuit fallback (standing in for RPP)
are rolled out ROLLOUT_TIME seconds on the costmap the policy sees. The policy
drives if its rollout is collision free and not much worse than the fallback's;
otherwise the fallback drives if it is safe; otherwise the car stops.
"""
import math

import numpy as np

from . import config as C
from .obs import FOOTPRINT_PTS

# same defaults as the plugin parameters in nav2_params.yaml
ROLLOUT_TIME = 2.0
ROLLOUT_DT = 0.1
NN_MARGIN = 0.5
COST_WEIGHT = 1.0
TAU_V, ACCEL_MAX, TAU_DELTA, STEER_RATE = 0.6, 2.5, 0.1, 1.0

NN, FALLBACK, STOP = 0, 1, 2
SOURCE_NAMES = ('NN', 'RPP', 'STOP')

# RPP fallback (FollowPath.fallback in nav2_params.yaml)
FALLBACK_SPEED = 3.0


def rollout(costmap, tracker, x, y, yaw, v, delta, v_cmd, d_cmd, s_now):
    """Returns (collides, score) for holding (v_cmd, d_cmd) ROLLOUT_TIME seconds."""
    steps = max(1, int(round(ROLLOUT_TIME / ROLLOUT_DT)))
    sub = 5
    h = ROLLOUT_DT / sub
    poses = np.empty((steps, 3))
    for i in range(steps):
        for _ in range(sub):
            v += max(-ACCEL_MAX * h, min(ACCEL_MAX * h, (v_cmd - v) * min(1.0, h / TAU_V)))
            delta += max(-STEER_RATE * h, min(STEER_RATE * h, (d_cmd - delta) * min(1.0, h / TAU_DELTA)))
            x += v * math.cos(yaw) * h
            y += v * math.sin(yaw) * h
            yaw += v * math.tan(delta) / C.WHEELBASE * h
        poses[i] = (x, y, yaw)
    # all footprint points of all rollout poses in one costmap lookup
    c, s = np.cos(poses[:, 2])[:, None], np.sin(poses[:, 2])[:, None]
    fx, fy = FOOTPRINT_PTS[:, 0], FOOTPRINT_PTS[:, 1]
    cost = costmap.cost_at(poses[:, 0:1] + c * fx - s * fy, poses[:, 1:2] + s * fx + c * fy)
    lethal = (cost >= C.LETHAL) & (cost != C.NO_INFORMATION)
    step_cost = np.where(cost == C.NO_INFORMATION, 0, cost).max(axis=1).astype(np.float64)
    if lethal.any():
        return True, 0.0  # the score of a colliding rollout is never used
    progress = tracker.peek(poses[-1, 0], poses[-1, 1])[2] - s_now
    return False, progress - COST_WEIGHT * step_cost.mean() / C.LETHAL


def pursuit_command(tracker, x, y, yaw, v, v_lim):
    """Regulated-pure-pursuit-like fallback: (v, delta) toward a point ahead on the path."""
    _, _, s_proj, _ = tracker.peek(x, y)
    ld = min(6.0, max(3.0, 1.5 * v))
    tx, ty = tracker.point_at(s_proj + ld)
    dx, dy = tx - x, ty - y
    by = -math.sin(yaw) * dx + math.cos(yaw) * dy
    delta = max(-C.MAX_STEER, min(C.MAX_STEER, math.atan2(2.0 * C.WHEELBASE * by, ld * ld)))
    curvature = abs(math.tan(delta)) / C.WHEELBASE
    v_cmd = FALLBACK_SPEED if curvature < 1.0 / 4.0 else FALLBACK_SPEED * (1.0 / curvature) / 4.0
    return min(v_cmd, v_lim), delta


def arbitrate(costmap, tracker, x, y, yaw, v, last_v, last_delta, nn_cmd, v_lim):
    """Returns ((v, delta), source). nn_cmd is already rate limited."""
    s_now = tracker.peek(x, y)[2]
    fb = C.rate_limit(*pursuit_command(tracker, x, y, yaw, v, v_lim), last_v, last_delta)
    fb_hit, fb_score = rollout(costmap, tracker, x, y, yaw, v, last_delta, fb[0], fb[1], s_now)
    nn_hit, nn_score = rollout(costmap, tracker, x, y, yaw, v, last_delta, nn_cmd[0], nn_cmd[1], s_now)
    if not nn_hit and (fb_hit or nn_score >= fb_score - NN_MARGIN):
        return nn_cmd, NN
    if not fb_hit:
        return fb, FALLBACK
    return (0.0, last_delta), STOP
