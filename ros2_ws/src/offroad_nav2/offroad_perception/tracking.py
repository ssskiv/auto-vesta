"""Temporal filter on a 2D hit-count grid and BEV clustering/tracking of class points. Pure numpy (+cv2 optional)."""
from collections import deque

import numpy as np


class TemporalFilter:
    """Keep a point only if its cell (cell m, in a fixed world frame) was hit in >= min_hits of the last `window`
    frames, current frame included. One instance per cloud."""

    def __init__(self, cell=0.2, window=3, min_hits=2):
        self.cell, self.min_hits = cell, min_hits
        self.hist = deque(maxlen=window)

    def _keys(self, xy):
        return np.floor(np.asarray(xy) / self.cell).astype(np.int64)

    def update(self, xy):
        """xy (N,2) world coords of this frame's points. Returns boolean keep mask (N,)."""
        if len(xy) == 0:
            self.hist.append(set())
            return np.zeros(0, bool)
        k = self._keys(xy)
        cur = set(map(tuple, np.unique(k, axis=0)))
        prev = list(self.hist)
        self.hist.append(cur)
        counts = {c: 1 for c in cur}
        for s in prev:
            for c in cur & s:
                counts[c] += 1
        good = {c for c, n in counts.items() if n >= self.min_hits}
        return np.fromiter((tuple(r) in good for r in k), bool, len(k))


def cluster_bev(xy, cell=0.3, min_pts=3, min_cells=2):
    """Connected components of occupied cells. Returns list of dicts {center(x,y), size(x,y), n}."""
    if len(xy) == 0:
        return []
    k = np.floor(xy / cell).astype(np.int64)
    k -= k.min(axis=0)
    shape = k.max(axis=0) + 3
    if shape.prod() > 4_000_000:
        return []
    grid = np.zeros(shape, np.uint8)
    grid[k[:, 0] + 1, k[:, 1] + 1] = 1
    import cv2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    li = lab[k[:, 0] + 1, k[:, 1] + 1]
    out = []
    for i in range(1, n):
        m = li == i
        if m.sum() < min_pts or stats[i, cv2.CC_STAT_AREA] < min_cells:
            continue
        p = xy[m]
        out.append({'center': p.mean(0), 'size': np.maximum(p.max(0) - p.min(0), cell), 'n': int(m.sum())})
    return out


class CentroidTracker:
    """Nearest-centroid association with stable integer ids."""

    def __init__(self, gate=1.5, max_age=5):
        self.gate, self.max_age, self.tracks, self.next_id = gate, max_age, {}, 1

    def update(self, clusters):
        used, res = set(), []
        for c in clusters:
            best, bd = None, self.gate
            for tid, t in self.tracks.items():
                d = float(np.linalg.norm(t['center'] - c['center']))
                if tid not in used and d < bd:
                    best, bd = tid, d
            if best is None:
                best = self.next_id
                self.next_id += 1
            used.add(best)
            self.tracks[best] = {'center': c['center'], 'age': 0}
            res.append((best, c))
        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]
        return res
