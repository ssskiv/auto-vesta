#!/usr/bin/env python3
"""Centreline lap route on the RoboCross static map: Dijkstra over free cells (cost ~ exp(-clearance)) through rough
via points, sampled every 18 m. Writes config/lap_route.json (map-frame x,y,yaw) + a PNG for inspection."""
import heapq
import json
import math
import sys

import cv2
import numpy as np

MAP = sys.argv[1]
OUT = sys.argv[2]
res, ox, oy = 0.2, -31.4, -113.0
im = cv2.imread(MAP, cv2.IMREAD_GRAYSCALE)
H, W = im.shape
free = (im >= 250).astype(np.uint8)
free = cv2.erode(free, np.ones((9, 9), np.uint8))          # keep 0.9 m from anything occupied/unknown
dt = cv2.distanceTransform(free, cv2.DIST_L2, 5) * res
# only the component containing the start: via points must not snap into disconnected strips
n_cc, lab = cv2.connectedComponents(free)
main_cc = lab[int(H - 1 - (-1 - -113.0) / res), int(round((0 + 31.4) / res))]
free = (lab == main_cc).astype(np.uint8)
print('main component cells', int(free.sum()), 'of', n_cc, 'components')
VIA = [(0, -1), (60, -1), (100, -3), (125, -18), (150, -45), (175, -80), (150, -98), (100, -78), (50, -58), (10, -42),
       (-3, -20), (0, -1)]
to_px = lambda x, y: (int(round((x - ox) / res)), int(round(H - 1 - (y - oy) / res)))
to_xy = lambda c, r: (ox + c * res, oy + (H - 1 - r) * res)


def snap(x, y):
    c, r = to_px(x, y)
    best, bd = (c, r), -1
    for dr in range(-60, 61):
        for dc in range(-60, 61):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and free[rr, cc] and dt[rr, cc] > bd:
                best, bd = (cc, rr), dt[rr, cc]
    return best


def dijkstra(a, b):
    cost = np.full((H, W), np.inf)
    prev = {}
    cost[a[1], a[0]] = 0
    pq = [(0.0, a)]
    while pq:
        d, (c, r) = heapq.heappop(pq)
        if (c, r) == b:
            break
        if d > cost[r, c]:
            continue
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            cc, rr = c + dc, r + dr
            if 0 <= cc < W and 0 <= rr < H and free[rr, cc]:
                nd = d + math.hypot(dc, dr) * (1.0 + 4.0 * math.exp(-dt[rr, cc] / 2.0))
                if nd < cost[rr, cc]:
                    cost[rr, cc] = nd
                    prev[(cc, rr)] = (c, r)
                    heapq.heappush(pq, (nd, (cc, rr)))
    p = [b]
    while p[-1] != a:
        p.append(prev[p[-1]])
    return p[::-1]


pts = [snap(*v) for v in VIA]
path = []
for a, b in zip(pts[:-1], pts[1:]):
    path += dijkstra(a, b)[:-1]
path.append(pts[-1])
xy = np.array([to_xy(c, r) for c, r in path])
wp = [xy[0]]
for p in xy:
    if np.hypot(*(p - wp[-1])) >= 18.0:
        wp.append(p)
route = []
for i, p in enumerate(wp):
    q = wp[(i + 1) % len(wp)]
    route.append({'x': round(float(p[0]), 2), 'y': round(float(p[1]), 2),
                  'yaw': round(math.atan2(q[1] - p[1], q[0] - p[0]), 3)})
json.dump(route, open(OUT, 'w'), indent=1)
img = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
for c, r in path[::3]:
    img[r, c] = (0, 0, 255)
for w in route:
    cv2.circle(img, to_px(w['x'], w['y']), 5, (255, 0, 0), -1)
cv2.imwrite(OUT.replace('.json', '.png'), img)
print(len(route), 'waypoints, path length %.0f m' % (len(path) * res * 1.1))
