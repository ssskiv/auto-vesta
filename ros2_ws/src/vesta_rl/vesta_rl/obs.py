"""Observation spec. This is the reference implementation.

vesta_neural_controller/src/observation.cpp mirrors it line by line; the C++
test compares both on fixtures written by make_obs_fixture.py.
"""
import math

import numpy as np

from . import config as C


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class CostmapView:
    """Nav2-style costmap: uint8 costs, data[my, mx], origin = corner of cell (0, 0)."""

    def __init__(self, data, origin_x, origin_y, resolution):
        self.data = data
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.resolution = resolution

    @property
    def size_x(self):
        return self.data.shape[1]

    @property
    def size_y(self):
        return self.data.shape[0]

    def cost_at(self, wx, wy):
        """Costs at world points (arrays); NO_INFORMATION outside the map."""
        mx = np.floor((np.asarray(wx) - self.origin_x) / self.resolution).astype(np.int64)
        my = np.floor((np.asarray(wy) - self.origin_y) / self.resolution).astype(np.int64)
        inside = (mx >= 0) & (mx < self.size_x) & (my >= 0) & (my < self.size_y)
        out = np.full(mx.shape, C.NO_INFORMATION, dtype=np.uint8)
        out[inside] = self.data[my[inside], mx[inside]]
        return out


def _patch_offsets():
    centers = C.PATCH_X_MIN + (np.arange(C.PATCH_SIZE) + 0.5) * C.PATCH_RES
    bx = centers[:, None].repeat(C.PATCH_SIZE, axis=1)          # row i -> x
    by = (C.PATCH_Y_MIN + (np.arange(C.PATCH_SIZE) + 0.5) * C.PATCH_RES)[None, :].repeat(C.PATCH_SIZE, axis=0)
    return bx, by


_PATCH_BX, _PATCH_BY = _patch_offsets()


def build_patch(costmap, px, py, yaw):
    """(PATCH_SIZE, PATCH_SIZE) float32 in [0, 1]; row i = x forward, col j = y left."""
    c, s = math.cos(yaw), math.sin(yaw)
    wx = px + c * _PATCH_BX - s * _PATCH_BY
    wy = py + s * _PATCH_BX + c * _PATCH_BY
    cost = costmap.cost_at(wx, wy)
    patch = cost.astype(np.float32) / float(C.LETHAL)
    patch[cost == C.NO_INFORMATION] = C.PATCH_UNKNOWN
    return patch


class PathTracker:
    """Projects the robot onto the global path, never jumping more than
    PATH_SEARCH_DIST ahead of the last match (keeps loops on the track sane)."""

    def __init__(self, path_xy):
        self.xy = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
        seg = np.diff(self.xy, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.s[-1])
        self.idx = 0

    def project(self, px, py):
        """Returns (proj_x, proj_y, s_proj, seg_heading) and advances idx."""
        n = len(self.xy)
        if n == 1:
            return self.xy[0, 0], self.xy[0, 1], 0.0, 0.0
        best_i, best_t, best_d2 = self.idx, 0.0, math.inf
        i = self.idx
        limit = self.s[self.idx] + C.PATH_SEARCH_DIST
        while i < n - 1 and self.s[i] <= limit:
            ax, ay = self.xy[i]
            bx, by = self.xy[i + 1]
            dx, dy = bx - ax, by - ay
            l2 = dx * dx + dy * dy
            t = 0.0 if l2 <= 0.0 else min(1.0, max(0.0, ((px - ax) * dx + (py - ay) * dy) / l2))
            qx, qy = ax + t * dx, ay + t * dy
            d2 = (px - qx) ** 2 + (py - qy) ** 2
            if d2 < best_d2:
                best_i, best_t, best_d2 = i, t, d2
            i += 1
        self.idx = best_i
        ax, ay = self.xy[best_i]
        bx, by = self.xy[best_i + 1]
        qx, qy = ax + best_t * (bx - ax), ay + best_t * (by - ay)
        s_proj = self.s[best_i] + best_t * self.seg_len[best_i]
        return qx, qy, s_proj, math.atan2(by - ay, bx - ax)

    def peek(self, px, py):
        """project() without moving idx (C++ PathTracker::peek)."""
        idx = self.idx
        out = self.project(px, py)
        self.idx = idx
        return out

    def point_at(self, s):
        if len(self.xy) == 1 or s >= self.length:
            return self.xy[-1, 0], self.xy[-1, 1]
        if s <= 0.0:
            return self.xy[0, 0], self.xy[0, 1]
        j = int(np.searchsorted(self.s, s, side='right')) - 1
        j = min(max(j, 0), len(self.xy) - 2)
        t = (s - self.s[j]) / self.seg_len[j] if self.seg_len[j] > 0.0 else 0.0
        return (self.xy[j, 0] + t * (self.xy[j + 1, 0] - self.xy[j, 0]),
                self.xy[j, 1] + t * (self.xy[j + 1, 1] - self.xy[j, 1]))


def to_base(px, py, yaw, x, y):
    dx, dy = x - px, y - py
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def build_vec(tracker, px, py, yaw, v, yaw_rate, last_v, last_delta, max_speed=C.MAX_SPEED):
    """(VEC_SIZE,) float32. Calls tracker.project, so call once per tick.

    Also returns the projection info so reward code doesn't project twice.
    """
    qx, qy, s_proj, seg_heading = tracker.project(px, py)
    out = np.zeros(C.VEC_SIZE, dtype=np.float32)
    for k in range(C.PATH_POINTS):
        x, y = tracker.point_at(s_proj + (k + 1) * C.PATH_SPACING)
        bx, by = to_base(px, py, yaw, x, y)
        out[2 * k] = bx / C.PATH_SCALE
        out[2 * k + 1] = by / C.PATH_SCALE
    o = 2 * C.PATH_POINTS
    out[o] = min(1.0, max(0.0, (tracker.length - s_proj) / C.GOAL_SCALE))
    _, cte = to_base(px, py, yaw, qx, qy)
    heading_err = wrap_angle(seg_heading - yaw)
    out[o + 1] = v / max_speed
    out[o + 2] = yaw_rate / C.YAW_RATE_SCALE
    out[o + 3] = last_v / max_speed
    out[o + 4] = last_delta / C.MAX_STEER
    out[o + 5] = min(1.0, max(-1.0, cte / C.CTE_SCALE))
    out[o + 6] = heading_err / math.pi
    return out, (s_proj, cte, heading_err)


def speed_limit(tracker, s, max_speed=C.MAX_SPEED, step=1.0):
    """Fastest speed at path position s from which every curve in the next
    C.LIMIT_HORIZON metres can be taken at <= C.A_LAT_MAX after braking at C.DECEL,
    and the end of the path reached at <= C.V_GOAL."""
    to_end = max(0.0, tracker.length - s)
    limit = min(max_speed, math.sqrt(C.V_GOAL ** 2 + 2.0 * C.DECEL * to_end))
    n = int(C.LIMIT_HORIZON / step) + 3
    # from 1 step behind the car: the heading change at pts[k] is (k - 1) * step ahead
    pts = [tracker.point_at(s + (k - 1) * step) for k in range(n)]
    prev_h = None
    for k in range(n - 1):
        (x0, y0), (x1, y1) = pts[k], pts[k + 1]
        if abs(x1 - x0) + abs(y1 - y0) < 1e-6:  # past the end of the path
            break
        h = math.atan2(y1 - y0, x1 - x0)
        if prev_h is not None:
            kappa = abs((h - prev_h + math.pi) % (2 * math.pi) - math.pi) / step
            v_curve = math.sqrt(C.A_LAT_MAX / kappa) if kappa > 1e-6 else max_speed
            limit = min(limit, math.sqrt(v_curve ** 2 + 2.0 * C.DECEL * (k - 1) * step))
        prev_h = h
    return limit


# Footprint sample points (base_link), used for collision and cost checks.
_FX = np.linspace(C.FOOTPRINT_X[0], C.FOOTPRINT_X[1], 24)
_FY = np.linspace(C.FOOTPRINT_Y[0], C.FOOTPRINT_Y[1], 10)
FOOTPRINT_PTS = np.stack(np.meshgrid(_FX, _FY, indexing='ij'), axis=-1).reshape(-1, 2)


def footprint_world(px, py, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    fx, fy = FOOTPRINT_PTS[:, 0], FOOTPRINT_PTS[:, 1]
    return px + c * fx - s * fy, py + s * fx + c * fy
