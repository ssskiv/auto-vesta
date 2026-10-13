"""Write observation test cases for the C++ test (vesta_neural_controller/test).

    ros2 run vesta_rl make_obs_fixture --out src/vesta_neural_controller/test/fixtures/obs_cases.txt

Rerun after changing obs.py or config.py.
"""
import argparse
import math

import numpy as np

from . import config as C
from .obs import CostmapView, PathTracker, build_patch, build_vec, speed_limit


def _case(rng, path_kind):
    sx, sy, res = int(rng.integers(60, 140)), int(rng.integers(60, 140)), float(rng.choice([0.05, 0.1, 0.2]))
    data = rng.integers(0, 256, (sy, sx), dtype=np.uint8)
    data[rng.random((sy, sx)) < 0.05] = C.NO_INFORMATION
    data[rng.random((sy, sx)) < 0.05] = C.LETHAL
    ox, oy = float(rng.uniform(-20, 0)), float(rng.uniform(-20, 0))
    cm = CostmapView(data, ox, oy, res)

    if path_kind == 'single':
        path = np.array([[rng.uniform(-5, 5), rng.uniform(-5, 5)]])
    else:
        n = 60 if path_kind == 'curve' else 8
        th = np.cumsum(rng.normal(0, 0.15, n)) + rng.uniform(-math.pi, math.pi)
        step = rng.uniform(0.3, 1.5, n)
        path = np.cumsum(np.stack([step * np.cos(th), step * np.sin(th)], axis=1), axis=0) + rng.uniform(-8, 0, 2)
        if path_kind == 'curve':
            path[10] = path[9]  # zero-length segment
    tracker = PathTracker(path)
    tracker.idx = int(rng.integers(0, max(1, len(path) - 1)) if path_kind == 'curve' and rng.random() < 0.5 else 0)
    idx_in = tracker.idx
    k = int(rng.integers(0, len(path)))
    px, py = path[k] + rng.normal(0, 1.0, 2)
    yaw = float(rng.uniform(-math.pi, math.pi))
    state = [float(rng.uniform(0, 5)), float(rng.normal(0, 0.3)), float(rng.uniform(0, 5)),
             float(rng.uniform(-0.6, 0.6)), float(rng.choice([C.MAX_SPEED, 3.0]))]
    vec, (s_proj, _, _) = build_vec(tracker, px, py, yaw, *state[:4], max_speed=state[4])
    patch = build_patch(cm, px, py, yaw)
    return dict(cm=cm, path=path, idx_in=idx_in, pose=(px, py, yaw), state=state,
                idx_out=tracker.idx, patch=patch, vec=vec, limit=speed_limit(tracker, s_proj, state[4]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    cases = [_case(rng, kind) for kind in ['curve'] * 6 + ['short', 'short', 'single']]

    f = lambda xs: ' '.join(f'{x:.9g}' for x in np.ravel(xs))  # noqa: E731
    f17 = lambda xs: ' '.join(f'{x:.17g}' for x in np.ravel(xs))  # noqa: E731
    with open(args.out, 'w') as out:
        out.write(f'{len(cases)}\n')
        for c in cases:
            cm = c['cm']
            out.write(f'{cm.size_x} {cm.size_y} {cm.origin_x:.9g} {cm.origin_y:.9g} {cm.resolution:.9g}\n')
            out.write(' '.join(map(str, cm.data.ravel())) + '\n')
            out.write(f"{len(c['path'])}\n{f17(c['path'])}\n")  # doubles: full precision
            out.write(f"{c['idx_in']} {f17(c['pose'])} {f17(c['state'])}\n")
            out.write(f"{c['idx_out']}\n{f(c['patch'])}\n{f(c['vec'])}\n{c['limit']:.17g}\n")
    print(f'wrote {len(cases)} cases to {args.out}')


if __name__ == '__main__':
    main()
