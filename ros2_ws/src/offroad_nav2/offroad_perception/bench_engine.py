"""p50/p95 latency of preprocess + three networks + argmax/CPU copy. Usage: python3 -m offroad_perception.bench_engine"""
import argparse
import os
import time

import numpy as np
import torch

from . import engine as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='/offroad/Models')
    ap.add_argument('--dir', default='/models')
    ap.add_argument('--n', type=int, default=200)
    a = ap.parse_args()
    eng = E.Engine(a.models, f'{a.dir}/last_freespaceseg.pth', f'{a.dir}/last_freespacecontour.pth',
                   f'{a.dir}/last_objectseg.pth')
    img = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    for _ in range(20):
        eng.infer(img)
    tot, nets = [], {'seg': [], 'contour': [], 'objseg': []}
    for _ in range(a.n):
        t0 = time.perf_counter()
        x = E.preprocess(img, eng.device, eng.half)
        out = eng.infer_tensor(x, sync=True)
        torch.cuda.synchronize()
        tot.append((time.perf_counter() - t0) * 1000)
        for k, v in out['timing_ms'].items():
            nets[k].append(v)
    p = lambda v: f'{np.percentile(v, 50):6.1f} / {np.percentile(v, 95):6.1f}'
    print(f'p50 / p95 ms over {a.n} frames (1280x720 input, fp16, GPU may be shared with training)')
    print('preprocess+3 nets+postprocess:', p(tot))
    for k, v in nets.items():
        print(f'  {k:8s}', p(v))


if __name__ == '__main__':
    main()
