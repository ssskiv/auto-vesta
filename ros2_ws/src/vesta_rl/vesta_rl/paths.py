"""Library of drivable global paths on the recorded map, generated once.

Grid A* with a clearance cost, then smoothing and a curvature check against
the car's minimum turning radius. Run:
    ros2 run vesta_rl make_paths --map <maps/robocross.yaml> --out <paths.npz>
"""
import argparse
import heapq
import math
import multiprocessing as mp

import numpy as np
from scipy.ndimage import distance_transform_edt

from .world import load_map

DOWNSAMPLE = 2           # planning grid = map resolution * 2
MIN_CLEARANCE = 2.0      # m; also closes gaps < 4 m between barrier objects
ENDPOINT_CLEARANCE = 2.5
MIN_DIST, MAX_DIST = 30.0, 150.0
MIN_TURN_RADIUS = 3.9
RESAMPLE = 0.5

_G = {}


def _init(map_yaml, min_clearance):
    global MIN_CLEARANCE
    MIN_CLEARANCE = min_clearance
    occupied, free, ox, oy, res = load_map(map_yaml)
    clearance = distance_transform_edt(free) * res
    h, w = clearance.shape
    h2, w2 = h // DOWNSAMPLE, w // DOWNSAMPLE
    ds = clearance[:h2 * DOWNSAMPLE, :w2 * DOWNSAMPLE].reshape(h2, DOWNSAMPLE, w2, DOWNSAMPLE).min(axis=(1, 3))
    _G.update(clearance=clearance, ds=ds, ox=ox, oy=oy, res=res, pres=res * DOWNSAMPLE)


def _astar(start, goal):
    ds = _G['ds']
    h, w = ds.shape
    allowed = ds >= MIN_CLEARANCE
    penalty = 1.0 + 3.0 * np.exp(-(ds - MIN_CLEARANCE) / 0.7)
    gx, gy = goal
    g = {start: 0.0}
    parent = {start: None}
    heap = [(math.hypot(start[0] - gx, start[1] - gy), 0.0, start)]
    moves = [(dx, dy, math.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    closed = set()
    while heap:
        _, gc, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        if cur == goal:
            out = []
            while cur is not None:
                out.append(cur)
                cur = parent[cur]
            return out[::-1]
        closed.add(cur)
        cx, cy = cur
        for dx, dy, l in moves:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h) or not allowed[ny, nx]:
                continue
            ng = gc + l * penalty[ny, nx]
            if ng < g.get((nx, ny), math.inf):
                g[(nx, ny)] = ng
                parent[(nx, ny)] = cur
                heapq.heappush(heap, (ng + math.hypot(nx - gx, ny - gy), ng, (nx, ny)))
    return None


def _clearance_at(x, y):
    c = _G['clearance']
    mx = int((x - _G['ox']) / _G['res'])
    my = int((y - _G['oy']) / _G['res'])
    if 0 <= mx < c.shape[1] and 0 <= my < c.shape[0]:
        return c[my, mx]
    return 0.0


def resample(xy, step):
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.arange(0.0, s[-1], step)
    t = np.append(t, s[-1])
    return np.stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])], axis=1)


def smooth(xy, iters=200, alpha=0.25):
    xy = xy.copy()
    for _ in range(iters):
        mid = xy[1:-1] + alpha * (xy[:-2] + xy[2:] - 2.0 * xy[1:-1])
        for i, (x, y) in enumerate(mid, start=1):
            if _clearance_at(x, y) >= MIN_CLEARANCE - 0.1:
                xy[i] = (x, y)
    return xy


def max_curvature(xy, window=3):
    d = np.diff(xy, axis=0)
    th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    seg = np.hypot(d[:, 0], d[:, 1])
    if len(th) <= window:
        return 0.0
    dth = np.abs(th[window:] - th[:-window])
    ds = np.convolve(seg, np.ones(window), 'valid')[:len(dth)]
    return float(np.max(dth / np.maximum(ds, 1e-6)))


def _one(seed):
    rng = np.random.default_rng(seed)
    ds = _G['ds']
    pres = _G['pres']
    ys, xs = np.nonzero(ds >= ENDPOINT_CLEARANCE)
    for _ in range(50):
        i, j = rng.integers(len(xs), size=2)
        d = math.hypot(xs[i] - xs[j], ys[i] - ys[j]) * pres
        if not MIN_DIST <= d <= MAX_DIST:
            continue
        cells = _astar((int(xs[i]), int(ys[i])), (int(xs[j]), int(ys[j])))
        if cells is None:
            continue
        xy = np.array([[_G['ox'] + (cx + 0.5) * pres, _G['oy'] + (cy + 0.5) * pres] for cx, cy in cells])
        xy = smooth(resample(xy, 1.0))
        xy = resample(xy, RESAMPLE)
        if max_curvature(xy) > 1.0 / MIN_TURN_RADIUS:
            continue
        return xy
    return None


def save_paths(path, paths):
    offsets = np.cumsum([0] + [len(p) for p in paths])
    np.savez_compressed(path, xy=np.concatenate(paths), offsets=offsets)


def load_paths(path):
    d = np.load(path)
    xy, off = d['xy'], d['offsets']
    return [xy[off[i]:off[i + 1]] for i in range(len(off) - 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--map', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min-clearance', type=float, default=MIN_CLEARANCE,
                    help='m from obstacles; 2.0 closes gaps < 4 m between barrier objects')
    args = ap.parse_args()

    with mp.Pool(args.workers, initializer=_init, initargs=(args.map, args.min_clearance)) as pool:
        paths = []
        seeds = iter(range(args.seed * 1_000_000, (args.seed + 1) * 1_000_000))
        while len(paths) < args.n:
            batch = [next(seeds) for _ in range(args.workers * 4)]
            paths += [p for p in pool.map(_one, batch) if p is not None]
            print(f'{min(len(paths), args.n)}/{args.n} paths', flush=True)
    paths = paths[:args.n]
    save_paths(args.out, paths)
    lengths = [float(np.sum(np.hypot(*np.diff(p, axis=0).T))) for p in paths]
    print(f'saved {len(paths)} paths to {args.out}, length {np.min(lengths):.0f}..{np.max(lengths):.0f} m')


if __name__ == '__main__':
    main()
