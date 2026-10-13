"""Can the real car get from the start of the path to its end? Finds the route.

Hybrid-A* with Dubins analytic expansion, like Nav2's SmacPlannerHybrid with
motion_model_for_search DUBIN (forward only):
  - motion primitives: forward arcs of ARC metres at curvatures 0, +-1/(2R), +-1/R;
  - every few expansions near the goal, a Dubins curve to the goal pose is tried,
    and the search ends when it is collision free;
  - heuristic: path length left from the nearest point of the global path.
Stricter than what the car can do on paper, so the found route is one the
policy can actually drive:
  - turning radius R = TURN_RADIUS (car: 3.9 m), since the steering needs time
    to turn and a Dubins curve switches curvature instantly;
  - the whole 4.6 x 1.8 m body keeps MARGIN metres from every obstacle;
  - base_link stays within CORRIDOR metres of the global path (further, the
    episode ends as "off path" at reward.CTE_LIMIT).
Used offline by scene_pool.py; training does no checks.
"""
import heapq
import math

import numpy as np

from . import config as C
from .world import cv2, distance_to_true

TURN_RADIUS = 5.0         # m
MARGIN = 1.0              # m between the body and any obstacle
CORRIDOR = 4.5            # m from the path for base_link; reward.CTE_LIMIT is 5
GOAL_TOL = 1.0            # m; reward.GOAL_TOL is 1.5
ARC = 1.0                 # m per motion primitive
STEP = 0.25               # m between collision-checked poses
CURVATURES = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) / TURN_RADIUS
CURVATURE_COST = 0.1      # extra cost per metre per unit of |curvature| * R: prefers straight driving
CELL = 0.5                # m, search grid for visited states
HEADING_BINS = 72
WEIGHT = 1.5              # weighted A*: much faster, slightly longer routes
MAX_EXPANSIONS = 30000
SHOT_DIST = 30.0          # m to the goal below which Dubins shots are tried ...
SHOT_EVERY = 5            # ... every this many expansions
ROUTE_STEP = 0.5          # m between points of the returned route

# body check points: the outline and the centre line (an obstacle inside the
# body is within half the width of the centre line), base_link frame
_P = 0.2
_xs = np.arange(C.FOOTPRINT_X[0], C.FOOTPRINT_X[1] + 1e-9, _P)
_ys = np.arange(C.FOOTPRINT_Y[0], C.FOOTPRINT_Y[1] + 1e-9, _P)
BODY = np.concatenate([np.stack([_xs, np.full_like(_xs, C.FOOTPRINT_Y[0])], 1),
                       np.stack([_xs, np.full_like(_xs, C.FOOTPRINT_Y[1])], 1),
                       np.stack([_xs, np.zeros_like(_xs)], 1),
                       np.stack([np.full_like(_ys, C.FOOTPRINT_X[0]), _ys], 1),
                       np.stack([np.full_like(_ys, C.FOOTPRINT_X[1]), _ys], 1)])


def _arc(x, y, th, k, s):
    """Poses after driving s (array) at curvature k from (x, y, th)."""
    s = np.asarray(s, float)
    if abs(k) < 1e-9:
        return np.stack([x + s * math.cos(th), y + s * math.sin(th), np.full_like(s, th)], -1)
    th1 = th + k * s
    return np.stack([x + (np.sin(th1) - math.sin(th)) / k, y - (np.cos(th1) - math.cos(th)) / k, th1], -1)


# ---- Dubins curves ---------------------------------------------------------
def _mod2pi(a):
    return a % (2 * math.pi)


def _dubins_words(a, b, d):
    """(word, t, p, q) for the 6 Dubins words; segment lengths in units of R."""
    sa, sb, ca, cb = math.sin(a), math.sin(b), math.cos(a), math.cos(b)
    cab = math.cos(a - b)
    out = []
    p2 = 2 + d * d - 2 * cab + 2 * d * (sa - sb)
    if p2 >= 0:
        tmp = math.atan2(cb - ca, d + sa - sb)
        out.append(('LSL', _mod2pi(-a + tmp), math.sqrt(p2), _mod2pi(b - tmp)))
    p2 = 2 + d * d - 2 * cab + 2 * d * (sb - sa)
    if p2 >= 0:
        tmp = math.atan2(ca - cb, d - sa + sb)
        out.append(('RSR', _mod2pi(a - tmp), math.sqrt(p2), _mod2pi(-b + tmp)))
    p2 = -2 + d * d + 2 * cab + 2 * d * (sa + sb)
    if p2 >= 0:
        p = math.sqrt(p2)
        tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
        out.append(('LSR', _mod2pi(-a + tmp), p, _mod2pi(-b + tmp)))
    p2 = d * d - 2 + 2 * cab - 2 * d * (sa + sb)
    if p2 >= 0:
        p = math.sqrt(p2)
        tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
        out.append(('RSL', _mod2pi(a - tmp), p, _mod2pi(b - tmp)))
    tmp = (6 - d * d + 2 * cab + 2 * d * (sa - sb)) / 8
    if abs(tmp) <= 1:
        p = _mod2pi(2 * math.pi - math.acos(tmp))
        t = _mod2pi(a - math.atan2(ca - cb, d - sa + sb) + p / 2)
        out.append(('RLR', t, p, _mod2pi(a - b - t + p)))
    tmp = (6 - d * d + 2 * cab + 2 * d * (sb - sa)) / 8
    if abs(tmp) <= 1:
        p = _mod2pi(2 * math.pi - math.acos(tmp))
        t = _mod2pi(-a - math.atan2(ca - cb, d + sa - sb) + p / 2)
        out.append(('LRL', t, p, _mod2pi(b - a - t + p)))
    return out


def dubins(q0, q1, radius, step=STEP):
    """Poses (n, 3) every `step` metres along the shortest Dubins curve q0 -> q1, end included."""
    dx, dy = q1[0] - q0[0], q1[1] - q0[1]
    d = math.hypot(dx, dy) / radius
    theta = math.atan2(dy, dx)
    words = _dubins_words(_mod2pi(q0[2] - theta), _mod2pi(q1[2] - theta), d)
    word, *lengths = min(words, key=lambda w: w[1] + w[2] + w[3])
    pose, out = tuple(q0), []
    for seg, ln in zip(word, lengths):
        k = {'L': 1.0 / radius, 'S': 0.0, 'R': -1.0 / radius}[seg]
        ln *= radius
        s = np.append(np.arange(step, ln, step), ln)
        poses = _arc(*pose, k, s)
        out.append(poses)
        pose = tuple(poses[-1])
    return np.concatenate(out)


# ---- search ----------------------------------------------------------------
def _primitives():
    s = np.append(np.arange(STEP, ARC, STEP), ARC)
    return np.stack([_arc(0.0, 0.0, 0.0, k, s) for k in CURVATURES])  # (n_curv, n_samples, 3)


PRIMS = _primitives()
PRIM_COST = ARC * (1.0 + CURVATURE_COST * np.abs(CURVATURES) * TURN_RADIUS)


class Feasibility:
    """Per map: corridor and heuristic are built once from the path; check()
    then searches a route for a given occupancy grid (static world + obstacles)."""

    def __init__(self, world, tracker):
        self.world, self.tracker = world, tracker
        res = world.res
        xy = tracker.xy
        pad = CORRIDOR + C.FOOTPRINT_X[1] + MARGIN + 2.0
        lo, hi = xy.min(axis=0) - pad, xy.max(axis=0) + pad
        (x0, y0), (x1, y1) = world.to_cell(lo[0], lo[1]), world.to_cell(hi[0], hi[1])
        self.x0, self.y0 = int(x0), int(y0)
        self.shape = (int(y1) + 1 - self.y0, int(x1) + 1 - self.x0)
        self.ox, self.oy = world.ox + self.x0 * res, world.oy + self.y0 * res

        # path raster with arc length per pixel -> distance and nearest-point arc length per cell
        dense_s = np.arange(0.0, tracker.length, res / 2)
        pts = np.array([tracker.point_at(s) for s in dense_s] + [tuple(xy[-1])])
        dense_s = np.append(dense_s, tracker.length)
        mx, my = self._cells(pts[:, 0], pts[:, 1])
        on_path = np.zeros(self.shape, dtype=bool)
        s_grid = np.zeros(self.shape)
        on_path[my, mx] = True
        s_grid[my, mx] = dense_s
        if cv2 is not None:
            dist, labels = cv2.distanceTransformWithLabels(
                (~on_path).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5, labelType=cv2.DIST_LABEL_PIXEL)
            s_near = np.concatenate([[0.0], s_grid[on_path]])[labels]  # labels: path pixels in raster order
        else:
            from scipy.ndimage import distance_transform_edt
            dist, (iy, ix) = distance_transform_edt(~on_path, return_indices=True)
            s_near = s_grid[iy, ix]
        self.corridor = dist * res <= CORRIDOR
        self.h = tracker.length - s_near  # metres of path left: works around hairpins too

        x, y = xy[0]
        nx, ny = xy[min(2, len(xy) - 1)]
        self.start = (float(x), float(y), math.atan2(ny - y, nx - x))
        px, py = xy[max(0, len(xy) - 3)]
        self.goal = (float(xy[-1, 0]), float(xy[-1, 1]), math.atan2(xy[-1, 1] - py, xy[-1, 0] - px))

    def _cells(self, x, y):
        mx = np.floor((np.asarray(x) - self.ox) / self.world.res).astype(np.int64)
        my = np.floor((np.asarray(y) - self.oy) / self.world.res).astype(np.int64)
        return np.clip(mx, 0, self.shape[1] - 1), np.clip(my, 0, self.shape[0] - 1)

    def _blocked(self, occupied):
        """Cells closer than MARGIN to an occupied cell (+ half a cell for its own extent)."""
        w = self.world
        win = np.zeros(self.shape, dtype=bool)  # outside the map = open ground
        sx0, sy0 = max(self.x0, 0), max(self.y0, 0)
        sx1, sy1 = min(self.x0 + self.shape[1], w.shape[1]), min(self.y0 + self.shape[0], w.shape[0])
        if sx1 > sx0 and sy1 > sy0:
            win[sy0 - self.y0:sy1 - self.y0, sx0 - self.x0:sx1 - self.x0] = occupied[sy0:sy1, sx0:sx1]
        if not win.any():
            return win
        return distance_to_true(win) * w.res < MARGIN + w.res / 2

    def check(self, occupied):
        """Returns the route (n, 2) every ROUTE_STEP metres from start to goal, or None."""
        blocked = self._blocked(occupied)

        def valid(poses):  # (..., 3) -> bool (...)
            c, s = np.cos(poses[..., 2])[..., None], np.sin(poses[..., 2])[..., None]
            bx = poses[..., 0:1] + c * BODY[:, 0] - s * BODY[:, 1]
            by = poses[..., 1:2] + s * BODY[:, 0] + c * BODY[:, 1]
            mx, my = self._cells(bx, by)
            px, py = self._cells(poses[..., 0], poses[..., 1])
            return ~blocked[my, mx].any(axis=-1) & self.corridor[py, px]

        x, y, th = self.start
        gx, gy, gth = self.goal
        if not valid(np.array([[x, y, th]]))[0]:
            return None
        bin_size = 2 * math.pi / HEADING_BINS

        def key(x, y, th):
            return int(x // CELL), int(y // CELL), int(round(th / bin_size)) % HEADING_BINS

        # node: (x, y, th, parent index, poses that led here)
        nodes = [(x, y, th, -1, np.array([[x, y, th]]))]
        mx, my = self._cells(x, y)
        heap = [(WEIGHT * self.h[my, mx], 0.0, 0)]
        closed = set()
        for n_exp in range(MAX_EXPANSIONS):
            if not heap:
                return None
            _, g, idx = heapq.heappop(heap)
            x, y, th, _, _ = nodes[idx]
            k = key(x, y, th)
            if k in closed:
                continue
            closed.add(k)
            dist_goal = math.hypot(x - gx, y - gy)
            if dist_goal <= GOAL_TOL:
                return self._route(nodes, idx, None)
            if dist_goal <= SHOT_DIST and n_exp % SHOT_EVERY == 0:
                shot = dubins((x, y, th), self.goal, TURN_RADIUS)
                if valid(shot).all():
                    return self._route(nodes, idx, shot)
            c, s = math.cos(th), math.sin(th)
            poses = np.empty(PRIMS.shape)
            poses[..., 0] = x + c * PRIMS[..., 0] - s * PRIMS[..., 1]
            poses[..., 1] = y + s * PRIMS[..., 0] + c * PRIMS[..., 1]
            poses[..., 2] = th + PRIMS[..., 2]
            ok = valid(poses).all(axis=1)
            ends = poses[:, -1]
            emx, emy = self._cells(ends[:, 0], ends[:, 1])
            hs = self.h[emy, emx]
            for i in np.nonzero(ok)[0]:
                ex, ey, eth = ends[i]
                if key(ex, ey, eth) in closed:
                    continue
                ng = g + PRIM_COST[i]
                nodes.append((ex, ey, eth, idx, poses[i]))
                heapq.heappush(heap, (ng + WEIGHT * hs[i], ng, len(nodes) - 1))
        return None

    def _route(self, nodes, idx, shot):
        parts = [] if shot is None else [shot]
        while idx >= 0:
            _, _, _, parent, poses = nodes[idx]
            parts.append(poses)
            idx = parent
        poses = np.concatenate(parts[::-1])[:, :2]
        # resample every ROUTE_STEP metres
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(poses, axis=0).T))])
        t = np.append(np.arange(0.0, s[-1], ROUTE_STEP), s[-1])
        return np.stack([np.interp(t, s, poses[:, 0]), np.interp(t, s, poses[:, 1])], 1)
