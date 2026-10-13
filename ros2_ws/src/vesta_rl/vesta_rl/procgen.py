"""Procedural worlds, so the policy does not get used to one track.

Each call makes a new random track: a smooth path the car can drive
(|curvature| < 1 / minimum turning radius), barriers along it in a random
style and width (dashed lines, rows of objects, one side only, none at all),
gaps in the barriers and clutter behind them. The path is the global path;
it keeps PATH_CLEARANCE from every barrier, like paths.py does on the real map.
"""
import math

import numpy as np
from .world import World, distance_to_true

RES = 0.2
MARGIN = 25.0              # m of world around the path's bounding box
PATH_STEP = 0.5            # m between path points (same as the path library)
MAX_CURVATURE = 1.0 / 4.5  # car: 1 / 3.9; keep a margin
PATH_CLEARANCE = 2.0       # m, same as paths.MIN_CLEARANCE


def random_path(rng, length):
    """Clothoid-like: curvature drifts toward random targets, straights in between."""
    n = int(length / PATH_STEP)
    x = y = 0.0
    heading = rng.uniform(-math.pi, math.pi)
    k = k_target = 0.0
    max_dk = rng.uniform(0.004, 0.015)  # curvature change per step: smooth vs. twisty tracks
    pts = [(x, y)]
    for _ in range(n):
        if rng.random() < 0.03:
            k_target = 0.0 if rng.random() < 0.35 else rng.uniform(-MAX_CURVATURE, MAX_CURVATURE)
        k += max(-max_dk, min(max_dk, k_target - k))
        heading += k * PATH_STEP
        x += math.cos(heading) * PATH_STEP
        y += math.sin(heading) * PATH_STEP
        pts.append((x, y))
    return np.array(pts)


def self_close(path, max_gap=9.0):
    """True if the path loops back on itself: two points more than 10 m apart
    along the path closer than min(0.4 * that distance, max_gap). Allows hairpins
    and parallel straights max_gap apart, rejects loops and crossings."""
    p = path[::4]  # every 2 m
    d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
    arc = np.abs(np.arange(len(p))[:, None] - np.arange(len(p))[None, :]) * 4 * PATH_STEP
    check = arc >= 10.0
    return bool((d[check] < np.minimum(0.4 * arc[check], max_gap)).any())


def _normals(path):
    d = np.gradient(path, axis=0)
    d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    return np.stack([-d[:, 1], d[:, 0]], axis=1)


def _smooth_noise(rng, n, amplitude, corr_steps):
    """Slowly varying noise along the path (for barrier distance)."""
    knots = rng.normal(0.0, amplitude, n // corr_steps + 2)
    return np.interp(np.arange(n), np.arange(len(knots)) * corr_steps, knots)


class _Canvas:
    def __init__(self, path):
        lo = path.min(axis=0) - MARGIN
        hi = path.max(axis=0) + MARGIN
        self.ox, self.oy = float(lo[0]), float(lo[1])
        w, h = int(math.ceil((hi[0] - lo[0]) / RES)), int(math.ceil((hi[1] - lo[1]) / RES))
        self.grid = np.zeros((h, w), dtype=bool)

    def put(self, x, y):
        mx = np.floor((np.asarray(x) - self.ox) / RES).astype(np.int64)
        my = np.floor((np.asarray(y) - self.oy) / RES).astype(np.int64)
        ok = (mx >= 0) & (mx < self.grid.shape[1]) & (my >= 0) & (my < self.grid.shape[0])
        self.grid[my[ok], mx[ok]] = True

    def box(self, cx, cy, lx, ly, th):
        u = np.arange(-lx / 2, lx / 2 + 1e-9, RES / 2)
        v = np.arange(-ly / 2, ly / 2 + 1e-9, RES / 2)
        uu, vv = np.meshgrid(u, v)
        self.put(cx + math.cos(th) * uu - math.sin(th) * vv, cy + math.sin(th) * uu + math.cos(th) * vv)

    def disc(self, cx, cy, r):
        u = np.arange(-r, r + 1e-9, RES / 2)
        uu, vv = np.meshgrid(u, u)
        m = uu ** 2 + vv ** 2 <= r ** 2
        self.put(cx + uu[m], cy + vv[m])


def _barrier(rng, canvas, path, normals, side, style):
    """One side of the track at a slowly varying distance from the path."""
    n = len(path)
    # >= 3 m: in a tight turn the front corner of the 4.6 m car swings ~2 m outside the path
    base = rng.uniform(3.5, 8.0)
    dist = np.clip(base + _smooth_noise(rng, n, rng.uniform(0.0, 1.5), int(rng.integers(20, 80))), 3.0, 10.0)
    line = path + normals * (side * dist)[:, None]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])

    if style == 'dashed':  # continuous pieces with gaps, like a tape or a fence
        thick = rng.uniform(0.2, 0.5)
        on, pos = True, 0.0
        while pos < s[-1]:
            seg = rng.uniform(2.0, 15.0) if on else rng.uniform(0.0, 4.0)
            if on:
                t = np.arange(pos, min(pos + seg, s[-1]), RES / 2)
                px, py = np.interp(t, s, line[:, 0]), np.interp(t, s, line[:, 1])
                nx, ny = np.interp(t, s, normals[:, 0]), np.interp(t, s, normals[:, 1])
                for o in np.arange(0.0, thick, RES / 2):
                    canvas.put(px + nx * side * o, py + ny * side * o)
            pos += seg
            on = not on
    else:  # 'objects': a row of tyres / boxes / X-shapes, like the robocross barriers
        spacing = rng.uniform(0.8, 4.0)
        size = rng.uniform(0.4, 1.5)
        for t in np.arange(rng.uniform(0, spacing), s[-1], spacing):
            if rng.random() < 0.1:
                continue  # a missing object = a gap
            cx, cy = np.interp(t, s, line[:, 0]), np.interp(t, s, line[:, 1])
            sz = size * rng.uniform(0.7, 1.3)
            kind = rng.integers(3)
            if kind == 0:
                canvas.disc(cx, cy, sz / 2)
            elif kind == 1:
                canvas.box(cx, cy, sz, sz * rng.uniform(0.3, 1.0), rng.uniform(0, math.pi))
            else:  # X
                th = rng.uniform(0, math.pi)
                canvas.box(cx, cy, sz * 1.4, 0.2, th)
                canvas.box(cx, cy, sz * 1.4, 0.2, th + math.pi / 2)


def _clutter(rng, canvas, path, density):
    """Random objects all over the world (filtered away from the path later)."""
    h, w = canvas.grid.shape
    n = rng.poisson(density * h * w * RES * RES)
    xs = canvas.ox + rng.uniform(0, w * RES, n)
    ys = canvas.oy + rng.uniform(0, h * RES, n)
    for x, y in zip(xs, ys):
        if rng.random() < 0.5:
            canvas.disc(x, y, rng.uniform(0.15, 0.8))
        else:
            canvas.box(x, y, rng.uniform(0.3, 3.0), rng.uniform(0.3, 3.0), rng.uniform(0, math.pi))


LAYOUTS = ('two_sides', 'one_side', 'open')
STYLES = ('dashed', 'objects')


def generate(rng, min_length=40.0, max_length=200.0, layout=None, style=None):
    """Returns (World, path_xy). layout / style force the barrier layout (LAYOUTS) and
    style (STYLES); by default they are random."""
    length = rng.uniform(min_length, max_length)
    while True:
        path = random_path(rng, length)
        if not self_close(path):
            break
        length = max(min_length, length * 0.9)  # long twisty paths loop more often
    canvas = _Canvas(path)
    normals = _normals(path)

    if layout is None:
        layout = rng.choice(LAYOUTS, p=[0.6, 0.2, 0.2])
    sides = {'two_sides': (1, -1), 'one_side': (int(rng.choice([1, -1])),), 'open': ()}[layout]
    for side in sides:
        _barrier(rng, canvas, path, normals, side, style or rng.choice(STYLES))
    _clutter(rng, canvas, path, density=rng.uniform(0.0, 0.03 if layout == 'open' else 0.01))

    # the global path keeps PATH_CLEARANCE from everything static; extended past
    # the goal (the nose is 3.7 m ahead of base_link) and behind the start
    def extend(p_end, p_prev, length):
        d = (p_end - p_prev) / max(np.hypot(*(p_end - p_prev)), 1e-9)
        return p_end + d[None, :] * np.arange(PATH_STEP, length + 1e-9, PATH_STEP)[:, None]
    full = np.concatenate([extend(path[0], path[1], 1.5)[::-1], path, extend(path[-1], path[-2], 5.0)])
    on_path = np.zeros_like(canvas.grid)
    dense = np.concatenate([np.linspace(a, b, 4, endpoint=False) for a, b in zip(full[:-1], full[1:])] + [full[-1:]])
    mx = np.floor((dense[:, 0] - canvas.ox) / RES).astype(np.int64)
    my = np.floor((dense[:, 1] - canvas.oy) / RES).astype(np.int64)
    on_path[my, mx] = True
    canvas.grid &= distance_to_true(on_path) * RES >= PATH_CLEARANCE

    return World.from_grid(canvas.grid, canvas.ox, canvas.oy, RES), path
