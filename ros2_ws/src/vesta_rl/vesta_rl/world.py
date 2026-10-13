"""2D world for the sim: the recorded map, lidar marking and Nav2-style inflation."""
import math
import os

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt

try:  # ~5x faster than scipy, exact with DIST_MASK_PRECISE
    import cv2
    cv2.setNumThreads(1)  # one thread per sim worker
except ImportError:
    cv2 = None

from . import config as C
from .obs import CostmapView


def load_map(yaml_path):
    """Returns (occupied, free, origin_x, origin_y, resolution); arrays are [my, mx]."""
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    image = meta['image']
    if not os.path.isabs(image):
        image = os.path.join(os.path.dirname(yaml_path), image)
    pix = read_pgm(image)
    if meta.get('negate', 0):
        pix = 255 - pix
    pix = np.flipud(pix)  # image row 0 is the top (max y); costmap row 0 is min y
    occ_p = (255.0 - pix) / 255.0
    occupied = occ_p > meta.get('occupied_thresh', 0.65)
    free = occ_p < meta.get('free_thresh', 0.25)
    ox, oy = meta['origin'][0], meta['origin'][1]
    return occupied, free, ox, oy, meta['resolution']


def read_pgm(path):
    with open(path, 'rb') as f:
        data = f.read()
    # header: magic, width, height, maxval, separated by whitespace and # comments
    tokens, pos = [], 0
    while len(tokens) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b'#':
            pos = data.index(b'\n', pos) + 1
            continue
        start = pos
        while not data[pos:pos + 1].isspace():
            pos += 1
        tokens.append(data[start:pos])
    assert tokens[0] == b'P5', 'only binary PGM is supported'
    w, h, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])
    assert maxval < 256
    pos += 1
    return np.frombuffer(data, dtype=np.uint8, count=w * h, offset=pos).reshape(h, w).copy()


def distance_to_true(mask):
    """Euclidean distance (cells) from every cell to the nearest True cell."""
    if cv2 is not None:
        return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return distance_transform_edt(~mask)


def connected_components(mask):
    """Labels of the 4-connected components of a boolean/uint8 mask (0 = background)."""
    if cv2 is not None:
        return cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)[1]
    from scipy.ndimage import label
    return label(mask)[0]


def inflate(lethal_mask, resolution):
    """Nav2 InflationLayer costs for a boolean lethal mask."""
    if not lethal_mask.any():
        return np.zeros(lethal_mask.shape, dtype=np.uint8)
    dist = distance_to_true(lethal_mask) * resolution
    cost = np.zeros(lethal_mask.shape, dtype=np.uint8)
    factor = np.exp(-C.COST_SCALING_FACTOR * (dist - C.INSCRIBED_RADIUS))
    ramp = ((C.INSCRIBED - 1) * factor).astype(np.uint8)
    band = (dist > C.INSCRIBED_RADIUS) & (dist <= C.INFLATION_RADIUS)
    cost[band] = ramp[band]
    cost[(dist <= C.INSCRIBED_RADIUS)] = C.INSCRIBED
    cost[lethal_mask] = C.LETHAL
    return cost


class World:
    """Static occupancy grid: the recorded map (read-only, shared) or a procedural one."""

    def __init__(self, map_yaml=None):
        if map_yaml is not None:
            self.occupied, self.free, self.ox, self.oy, self.res = load_map(map_yaml)
            self.shape = self.occupied.shape

    @classmethod
    def from_grid(cls, occupied, origin_x, origin_y, resolution):
        w = cls()
        w.occupied, w.free = occupied, ~occupied
        w.ox, w.oy, w.res = origin_x, origin_y, resolution
        w.shape = occupied.shape
        return w

    def to_cell(self, x, y):
        mx = np.floor((np.asarray(x) - self.ox) / self.res).astype(np.int64)
        my = np.floor((np.asarray(y) - self.oy) / self.res).astype(np.int64)
        return mx, my

    def inside(self, mx, my):
        return (mx >= 0) & (mx < self.shape[1]) & (my >= 0) & (my < self.shape[0])


class LocalCostmap:
    """What Nav2's local costmap would show: lidar-marked obstacles + inflation.

    Marks persist (static world). The inflated costs are kept for a window
    around the robot and recomputed only when new cells get marked or the
    robot leaves the middle of the window.
    """

    N_RAYS = 720
    RAY_STEP = 0.1
    WINDOW = 48.0
    RECENTER = 8.0

    def __init__(self, world, occupied, rng, drop_prob=0.0):
        self.world = world
        self.occupied = occupied          # per-episode truth: map + boxes
        self.seen = np.zeros(world.shape, dtype=bool)
        self.rng = rng
        self.drop_prob = drop_prob
        self.view = None
        self._center = None
        self._dirty = True
        ang = np.linspace(-math.pi, math.pi, self.N_RAYS, endpoint=False)
        r = np.arange(self.RAY_STEP, C.LIDAR_RANGE, self.RAY_STEP)
        self._ray_dx = np.cos(ang)[:, None] * r[None, :]
        self._ray_dy = np.sin(ang)[:, None] * r[None, :]

    def add_ghosts(self, xs, ys):
        mx, my = self.world.to_cell(xs, ys)
        ok = self.world.inside(mx, my)
        self.seen[my[ok], mx[ok]] = True
        self._dirty = True

    def add_ghost_cells(self, mx, my):
        self.seen[my, mx] = True
        self._dirty = True

    def scan(self, px, py, yaw):
        lx = px + C.LIDAR_X * math.cos(yaw)
        ly = py + C.LIDAR_X * math.sin(yaw)
        mx, my = self.world.to_cell(lx + self._ray_dx, ly + self._ray_dy)
        inside = self.world.inside(mx, my)
        hit = np.zeros(mx.shape, dtype=bool)
        hit[inside] = self.occupied[my[inside], mx[inside]]
        any_hit = hit.any(axis=1)
        if self.drop_prob > 0.0:
            any_hit &= self.rng.random(self.N_RAYS) >= self.drop_prob
        first = hit.argmax(axis=1)
        rows = np.nonzero(any_hit)[0]
        hx, hy = mx[rows, first[rows]], my[rows, first[rows]]
        new = ~self.seen[hy, hx]
        if new.any():
            self.seen[hy[new], hx[new]] = True
            self._dirty = True
        self._update(px, py)

    def _update(self, px, py):
        if (not self._dirty and self._center is not None
                and abs(px - self._center[0]) < self.RECENTER and abs(py - self._center[1]) < self.RECENTER):
            return
        w = self.world
        half = int(self.WINDOW / 2 / w.res)
        cx, cy = w.to_cell(px, py)
        x0, y0 = int(cx) - half, int(cy) - half
        size = 2 * half
        # copy the window (zero-padded outside the map) and inflate it
        window = np.zeros((size, size), dtype=bool)
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x0 + size, w.shape[1]), min(y0 + size, w.shape[0])
        if sx1 > sx0 and sy1 > sy0:
            window[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = self.seen[sy0:sy1, sx0:sx1]
        cost = inflate(window, w.res)
        self.view = CostmapView(cost, w.ox + x0 * w.res, w.oy + y0 * w.res, w.res)
        self._center = (px, py)
        self._dirty = False
