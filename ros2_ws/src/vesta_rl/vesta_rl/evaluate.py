"""Deterministic evaluation on fixed scenes of a pool, with the arbiter in the loop.

Two disjoint scene sets are drawn once per pool (seeded): "control" drives the
training supervisor's decisions (keep best, roll back), "holdout" is only
reported, so the supervisor cannot overfit the number it is judged by.

    ros2 run vesta_rl evaluate --pool src/vesta_rl/data/pool --model runs/v6/best.zip
    ros2 run vesta_rl evaluate --pool src/vesta_rl/data/pool --audit   # expert on every scene

--audit drives the BC expert (planner route) with the arbiter over every scene
and adds the ones it cannot finish to rejected.txt: if the arbiter blocks even
the planner's route, the scene only teaches the policy to get stuck.
"""
import argparse
import collections
import json
import multiprocessing as mp
import os

import numpy as np

PER_KIND = 10
SETS = {'control': 1, 'holdout': 2}  # seeds


def scene_sets(pool, per_kind=PER_KIND):
    """{'control': [(kind, i, j)], 'holdout': [...]}, disjoint, per_kind scenes of every kind."""
    by_kind = collections.defaultdict(list)
    for i, js in pool.choices:
        by_kind[pool.kind(i)].extend((i, int(j)) for j in js)
    rng = np.random.default_rng(12345)
    out = {name: [] for name in SETS}
    for kind in sorted(by_kind):
        scenes = by_kind[kind]
        pick = rng.choice(len(scenes), size=min(len(scenes), 2 * per_kind), replace=False)
        for n, name in enumerate(SETS):
            out[name] += [(kind, *scenes[k]) for k in pick[n * per_kind:(n + 1) * per_kind]]
    return out


_W = {}


def _init(pool_dir, model_path):
    import torch
    torch.set_num_threads(1)
    os.nice(10)
    from .scene_pool import ScenePool
    from .sim2d import Sim2DEnv
    _W['env'] = Sim2DEnv(ScenePool(pool_dir), seed=0)
    _W['env'].P_NO_OBSTACLES = 0.0
    _W['model'] = None
    if model_path != 'expert':
        from stable_baselines3 import PPO
        _W['model'] = PPO.load(model_path, device='cpu')


def _episode(task):
    k, kind, i, j = task
    env, model = _W['env'], _W['model']
    o, _ = env.reset(seed=10_000 + k, options={'scene': (i, j)})
    done, steps, stop = False, 0, 0
    while not done:
        a = env.expert_action() if model is None else model.predict(o, deterministic=True)[0]
        o, _, te, tr, info = env.step(a)
        done = te or tr
        steps += 1
        stop += info['source'] == 2
    return dict(kind=kind, i=i, j=j, reason=info['reason'], stop=stop / steps)


def run(pool_dir, model_path, scenes, workers):
    """Results per scene, in order."""
    tasks = [(k, *s) for k, s in enumerate(scenes)]
    with mp.get_context('spawn').Pool(workers, _init, (pool_dir, model_path)) as p:
        return p.map(_episode, tasks, chunksize=1)


def summary(results):
    reasons = collections.Counter(r['reason'] for r in results)
    kinds = collections.defaultdict(list)
    for r in results:
        kinds[r['kind']].append(r['reason'] == 'goal')
    return dict(success=reasons['goal'] / max(1, len(results)), reasons=dict(reasons),
                by_kind={k: float(np.mean(v)) for k, v in sorted(kinds.items())})


def evaluate(pool_dir, model_path, workers=4, per_kind=PER_KIND):
    from .scene_pool import ScenePool
    sets = scene_sets(ScenePool(pool_dir), per_kind)
    names = list(sets)
    results = run(pool_dir, model_path, [s for n in names for s in sets[n]], workers)
    out, k = {}, 0
    for n in names:
        out[n] = summary(results[k:k + len(sets[n])])
        k += len(sets[n])
    return out


def audit(pool_dir, workers):
    from .scene_pool import ScenePool, write_rejected
    pool = ScenePool(pool_dir)
    scenes = [(pool.kind(i), i, int(j)) for i, js in pool.choices for j in js]
    print(f'driving the expert with the arbiter over {len(scenes)} scenes on {workers} workers', flush=True)
    tasks = [(k, *s) for k, s in enumerate(scenes)]
    bad = []
    with mp.get_context('spawn').Pool(workers, _init, (pool_dir, 'expert')) as p:
        for n, r in enumerate(p.imap_unordered(_episode, tasks, chunksize=8), 1):
            if r['reason'] != 'goal':
                bad.append(r)
            if n % 1000 == 0:
                print(f'{n}/{len(scenes)}: {len(bad)} failed', flush=True)
    pool.reload_rejected()
    for r in bad:
        pool.rejected_layouts[r['i']].add(r['j'])
    write_rejected(pool_dir, pool.rejected_maps, pool.rejected_layouts)
    with open(os.path.join(pool_dir, 'audit.json'), 'w') as f:
        json.dump(bad, f)
    by = collections.Counter((r['kind'], r['reason']) for r in bad)
    for (kind, reason), n in sorted(by.items()):
        print(f'{kind:18s} {reason:10s} {n}')
    print(f'rejected {len(bad)} of {len(scenes)} scenes (details in audit.json)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pool', required=True)
    ap.add_argument('--model', help='model.zip, or "expert"')
    ap.add_argument('--audit', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--per-kind', type=int, default=PER_KIND)
    ap.add_argument('--out', help='write the result as JSON')
    args = ap.parse_args()
    if args.audit:
        audit(args.pool, args.workers)
        return
    res = evaluate(args.pool, args.model, args.workers, args.per_kind)
    print(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(res, f)


if __name__ == '__main__':
    main()
