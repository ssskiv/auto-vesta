"""PPO training in the 2D sim, optional BC warm start, fine-tuning in Webots.

    # bulk training in the 2D sim, on a reviewed scene pool (scene_pool.py)
    ros2 run vesta_rl train --pool <pool dir> --out runs/sim --steps 30000000
    # fine-tuning in Webots (robot.launch.py + nav2.launch.py running)
    ros2 run vesta_rl train --env webots --paths <paths.npz> \
        --resume runs/sim/model.zip --out runs/webots --steps 300000

Resource use: --n-envs worker processes (one CPU core each, ~200 MB each),
--threads for torch in the main process, all at --nice priority. --max-ram-gb
saves the model and stops before the machine starts swapping.
"""
import argparse
import collections
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .arbiter import SOURCE_NAMES
from .policy import POLICY_KWARGS, VestaPolicy, VestaSquashedPolicy


THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def process_tree_ram_gb():
    """Memory of this process and all its descendants, from /proc. Uses PSS
    (shared libraries like libtorch counted once), falling back to RSS."""
    parents = {}
    for pid in os.listdir('/proc'):
        if pid.isdigit():
            try:
                with open(f'/proc/{pid}/stat') as f:
                    parents[int(pid)] = int(f.read().rsplit(')', 1)[1].split()[1])
            except OSError:
                pass
    tree, frontier = {os.getpid()}, [os.getpid()]
    while frontier:
        p = frontier.pop()
        for child, parent in parents.items():
            if parent == p and child not in tree:
                tree.add(child)
                frontier.append(child)
    total_kb = 0
    for pid in tree:
        for path, key in ((f'/proc/{pid}/smaps_rollup', 'Pss:'), (f'/proc/{pid}/status', 'VmRSS:')):
            try:
                with open(path) as f:
                    kb = next((int(line.split()[1]) for line in f if line.startswith(key)), None)
            except OSError:
                kb = None
            if kb is not None:
                total_kb += kb
                break
    return total_kb / 1e6


class RamGuard(BaseCallback):
    """Stops training (the model is still saved) when the process tree exceeds max_gb."""

    def __init__(self, max_gb):
        super().__init__()
        self.max_gb = max_gb
        self.over = False

    def _on_rollout_end(self):
        used = process_tree_ram_gb()
        self.logger.record('system/ram_gb', used)
        if used > self.max_gb:
            print(f'RAM {used:.1f} GB > --max-ram-gb {self.max_gb}: stopping', file=sys.stderr)
            self.over = True

    def _on_step(self):
        return not self.over


class Curriculum(BaseCallback):
    """Raises the sim difficulty (obstacle count, share on the path) by `step` each
    time the success rate over the last 100 episodes reaches `up`, at most once per
    `min_steps`. The level is kept in <out>/curriculum.json across restarts."""

    def __init__(self, path, level, up=0.6, step=0.1, min_steps=300_000):
        super().__init__()
        self.path, self.level, self.up, self.step, self.min_steps = path, level, up, step, min_steps
        self.last_change = None

    def _apply(self):
        n = self.training_env.env_method('set_difficulty', self.level)[0]
        with open(self.path, 'w') as f:
            json.dump({'level': self.level}, f)
        print(f'curriculum level {self.level:.1f}: up to {n} obstacles', flush=True)

    def _on_training_start(self):
        self.last_change = self.num_timesteps
        self._apply()

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        buf = self.model.ep_success_buffer
        self.logger.record('curriculum/level', self.level)
        if (self.level < 1.0 and len(buf) >= 50 and np.mean(buf) >= self.up
                and self.num_timesteps - self.last_change >= self.min_steps):
            self.level = min(1.0, round(self.level + self.step, 3))
            self.last_change = self.num_timesteps
            self._apply()


class ActionStats(BaseCallback):
    """Logs how often the policy's raw throttle/steer actions hit the [-1, 1] bound."""

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        a = self.model.rollout_buffer.actions.reshape(-1, 2)
        # speed command as a fraction of the path speed limit (1 = at the limit)
        self.logger.record('actions/speed_frac_mean', float(np.mean((np.clip(a[:, 0], -1, 1) + 1) / 2)))
        # share of actions at the bound: clipped (VestaPolicy) or squashed into tanh's flat end
        self.logger.record('actions/throttle_saturated', float(np.mean(np.abs(a[:, 0]) > 0.99)))
        self.logger.record('actions/steer_saturated', float(np.mean(np.abs(a[:, 1]) > 0.99)))


class ShieldStats(BaseCallback):
    """Logs how often the arbiter (NN / RPP / STOP) let the policy drive."""

    def __init__(self):
        super().__init__()
        self.counts = np.zeros(len(SOURCE_NAMES))

    def _on_step(self):
        for info in self.locals['infos']:
            if 'source' in info:
                self.counts[info['source']] += 1
        return True

    def _on_rollout_end(self):
        n = self.counts.sum()
        if n:
            for i, name in enumerate(SOURCE_NAMES):
                self.logger.record(f'shield/{name.lower()}_share', float(self.counts[i] / n))
        self.counts[:] = 0


class Supervisor(BaseCallback):
    """Watches the training and steers it (sim2d only).

    Every `every` steps the current model is saved and evaluated deterministically,
    with the arbiter, on fixed scenes of the pool (evaluate.py) in a background
    process, so training does not wait. When a result arrives:
      - better than the best "control" success so far: kept as <out>/best.zip
        and exported to <out>/best.onnx;
      - `patience` results in a row more than `drop` below the best: training
        rolls back to best.zip and the learning rate is halved;
    "holdout" success is only logged: decisions never see those scenes.
    After every rollout, map kinds are sampled by how often the policy fails on
    them in training (half by failure rate, half as in the pool).
    State is kept in <out>/supervisor.json across restarts.
    """

    def __init__(self, out, pool_dir, every=1_000_000, workers=6, drop=0.10, patience=2, lr_min=1e-5,
                 kind_mix=0.5, window=200, per_kind=10):
        super().__init__()
        self.out, self.pool_dir, self.every, self.workers = out, pool_dir, every, workers
        self.per_kind = per_kind
        self.drop, self.patience, self.lr_min, self.kind_mix = drop, patience, lr_min, kind_mix
        self.eval_dir = os.path.join(out, 'supervisor')
        self.state_path = os.path.join(out, 'supervisor.json')
        self.state = dict(best=-1.0, best_step=None, holdout_at_best=None, bad=0, rollbacks=0, lr=None, history=[])
        self.pending = None
        self.next_eval = None
        self.outcomes = collections.defaultdict(lambda: collections.deque(maxlen=window))

    def _on_training_start(self):
        from .scene_pool import ScenePool
        os.makedirs(self.eval_dir, exist_ok=True)
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                self.state.update(json.load(f))
        if self.state['lr'] is not None:
            self._set_lr(self.state['lr'])
        else:
            self.state['lr'] = float(self.model.lr_schedule(1.0))
        pool = ScenePool(self.pool_dir)
        counts = collections.Counter(pool.kind(i) for i, _ in pool.choices)
        self.base = {k: n / sum(counts.values()) for k, n in counts.items()}
        self.next_eval = (self.num_timesteps // self.every + 1) * self.every

    def _set_lr(self, lr):
        self.model.lr_schedule = lambda _: lr
        self.model.learning_rate = lr
        for g in self.model.policy.optimizer.param_groups:
            g['lr'] = lr

    def _on_step(self):
        for info in self.locals['infos']:
            if 'kind' in info:
                self.outcomes[info['kind']].append(bool(info.get('is_success')))
        return True

    def _on_rollout_end(self):
        self._kind_weights()
        if self.pending is not None and self.pending[0].poll() is not None:
            self._act(*self.pending[1:])
            self.pending = None
        if self.pending is None and self.num_timesteps >= self.next_eval:
            step = self.num_timesteps
            path = os.path.join(self.eval_dir, f'candidate_{step}.zip')
            self.model.save(path)
            result = os.path.join(self.eval_dir, f'eval_{step}.json')
            cmd = [sys.executable, '-m', 'vesta_rl.evaluate', '--pool', self.pool_dir, '--model', path,
                   '--workers', str(self.workers), '--per-kind', str(self.per_kind), '--out', result]
            log = open(os.path.join(self.eval_dir, f'eval_{step}.log'), 'w')
            self.pending = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), step, path, result)
            self.next_eval = (step // self.every + 1) * self.every
        st = self.state
        self.logger.record('supervisor/best_control', max(0.0, st['best']))
        self.logger.record('supervisor/lr', st['lr'])
        self.logger.record('supervisor/rollbacks', st['rollbacks'])

    def _kind_weights(self):
        fail = {k: 1.0 - float(np.mean(self.outcomes[k])) if len(self.outcomes[k]) >= 20 else None
                for k in self.base}
        known = {k: f for k, f in fail.items() if f is not None}
        if len(known) < len(self.base) or sum(known.values()) <= 0:
            return
        total = sum(known.values())
        w = {k: (1 - self.kind_mix) * self.base[k] + self.kind_mix * known[k] / total for k in self.base}
        self.training_env.env_method('set_kind_weights', w)
        for k, v in w.items():
            self.logger.record(f'kind_weight/{k}', v)
            self.logger.record(f'kind_success/{k}', 1.0 - known[k])

    def _act(self, step, path, result):
        st = self.state
        try:
            with open(result) as f:
                res = json.load(f)
        except (OSError, ValueError):
            print(f'supervisor: evaluation at {step} failed, see {result[:-5]}.log', flush=True)
            return
        ctl, hold = res['control']['success'], res['holdout']['success']
        self.logger.record('supervisor/control_success', ctl)
        self.logger.record('supervisor/holdout_success', hold)
        for k, v in res['control']['by_kind'].items():
            self.logger.record(f'supervisor_kind/{k}', v)
        action = 'keep going'
        if ctl > st['best']:
            st.update(best=ctl, best_step=step, holdout_at_best=hold, bad=0)
            shutil.copy(path, os.path.join(self.out, 'best.zip'))
            try:
                from .export_onnx import export
                export(path, os.path.join(self.out, 'best.onnx'))
                action = 'new best, exported best.onnx'
            except Exception as e:  # noqa: BLE001 - keep training if the export breaks
                action = f'new best (ONNX export failed: {e})'
        elif ctl < st['best'] - self.drop:
            st['bad'] += 1
            action = f'below best ({st["bad"]}/{self.patience})'
            if st['bad'] >= self.patience:
                self.model.set_parameters(os.path.join(self.out, 'best.zip'), exact_match=True, device=self.model.device)
                st['lr'] = max(self.lr_min, st['lr'] * 0.5)
                self._set_lr(st['lr'])
                st['bad'] = 0
                st['rollbacks'] += 1
                action = f'ROLLBACK to best ({st["best_step"]}), lr -> {st["lr"]:.2e}'
        else:
            st['bad'] = 0
        st['history'].append(dict(step=step, control=ctl, holdout=hold, reasons=res['control']['reasons'],
                                  action=action))
        with open(self.state_path, 'w') as f:
            json.dump(st, f, indent=1)
        os.remove(path)
        print(f'supervisor @ {step}: control {ctl:.0%} holdout {hold:.0%} (best {st["best"]:.0%} @ {st["best_step"]}): '
              f'{action}', flush=True)


def bc_warm_start(model, env, n_steps, epochs, noise=0.15, seed=0):
    """Pretrain the actor mean on the pure-pursuit expert (DART-style noise:
    execute a noisy expert action, label with the clean one)."""
    rng = np.random.default_rng(seed)
    obs = env.reset()
    buf_cm, buf_vec, buf_act = [], [], []
    while len(buf_act) * env.num_envs < n_steps:
        expert = np.stack(env.env_method('expert_action'))
        buf_cm.append(obs['costmap'].astype(np.float16))  # cost/254 fits fp16; halves the RAM
        buf_vec.append(obs['vec'])
        buf_act.append(expert)
        act = np.clip(expert + rng.normal(0.0, noise, expert.shape), -1.0, 1.0).astype(np.float32)
        obs, _, _, _ = env.step(act)
    cm = np.concatenate(buf_cm)
    vec = np.concatenate(buf_vec)
    act = np.concatenate(buf_act)
    del buf_cm, buf_vec, buf_act
    print(f'BC: {len(act)} expert samples ({cm.nbytes / 1e9:.1f} GB), {epochs} epochs')

    def batch(a, idx, dtype=torch.float32):
        return torch.as_tensor(a[idx], device=model.device, dtype=dtype)

    policy = model.policy
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    n = len(act)
    for epoch in range(epochs):
        perm = rng.permutation(n)
        total = 0.0
        for i in range(0, n, 256):
            idx = np.sort(perm[i:i + 256])
            dist = policy.get_distribution({'costmap': batch(cm, idx), 'vec': batch(vec, idx)})
            mean = dist.distribution.mean
            if isinstance(policy, VestaSquashedPolicy):  # its Normal is over u, the action is tanh(u)
                mean = torch.tanh(mean)
            loss = torch.nn.functional.mse_loss(mean, batch(act, idx))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f'BC epoch {epoch + 1}/{epochs}: mse {total / n:.4f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--env', choices=['sim2d', 'webots'], default='sim2d')
    ap.add_argument('--pool', help='scene pool dir (scene_pool generate + review), for --env sim2d')
    ap.add_argument('--paths', help='path library, for --env webots')
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=30_000_000)
    ap.add_argument('--n-envs', type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)),
                    help='2D sim worker processes, one CPU core each (default: half the cores, max 8)')
    ap.add_argument('--threads', type=int, default=4, help='torch CPU threads in the main process')
    ap.add_argument('--nice', type=int, default=10, help='CPU priority (0 = normal, 19 = lowest)')
    ap.add_argument('--max-ram-gb', type=float, default=0.0,
                    help='save and stop when training uses more RAM than this (0 = off)')
    ap.add_argument('--max-obstacles', '--max-boxes', dest='max_obstacles', type=int, default=15,
                    help='at most this many of a scene\'s obstacles per episode (the curriculum starts at 3)')
    ap.add_argument('--resume', help='model.zip to continue from')
    ap.add_argument('--bc-steps', type=int, default=0, help='expert samples for the BC warm start (0 = off)')
    ap.add_argument('--bc-epochs', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-shield', dest='shield', action='store_false',
                    help='train without the arbiter (NN / RPP / STOP) in the loop')
    ap.add_argument('--policy', choices=['squashed', 'tanh-mean'], default='squashed',
                    help='squashed: a = tanh(u), u ~ N (default); tanh-mean: old VestaPolicy')
    ap.add_argument('--name', default=None,
                    help='run name for tensorboard and checkpoint files (default: env name / "ppo")')
    ap.add_argument('--ent-coef', type=float, default=0.003,
                    help='entropy bonus: keeps the policy trying different actions')
    ap.add_argument('--curriculum', type=float, default=None,
                    help='start difficulty 0..1 (0 = 3 obstacles, rarely on the path); default: the level saved '
                         'in <out>/curriculum.json, or 0 for a new run. 1 disables the curriculum')
    ap.add_argument('--rescale-action', type=float, nargs=2, metavar=('THROTTLE', 'STEER'),
                    help='with --resume: multiply the policy output layer rows (once, to pull a saturated '
                         'throttle back into tanh range, e.g. 0.22 1.0)')
    ap.add_argument('--reset-std', type=float, default=0.0,
                    help='with --resume: set the action std back to this (e.g. 0.55) to explore again')
    ap.add_argument('--target-kl', type=float, default=0.02,
                    help='PPO stops the epochs of an update when the policy moved further than this (0 = off)')
    ap.add_argument('--no-supervisor', dest='supervisor', action='store_false',
                    help='no periodic evaluation / best checkpoint / rollback (see Supervisor)')
    ap.add_argument('--eval-every', type=int, default=1_000_000)
    ap.add_argument('--eval-per-kind', type=int, default=10, help='scenes per map kind in each evaluation set')
    ap.add_argument('--eval-workers', type=int, default=6)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()
    if args.env == 'sim2d' and not args.pool:
        ap.error('--env sim2d needs --pool (generate and review one with scene_pool first)')
    if args.env == 'webots' and not args.paths:
        ap.error('--env webots needs --paths')
    os.makedirs(args.out, exist_ok=True)

    # Before the workers start, so they inherit it: one BLAS/OpenMP thread per
    # worker (otherwise every worker starts one thread per core).
    for var in THREAD_ENV:
        os.environ.setdefault(var, '1')
    torch.set_num_threads(args.threads)
    if args.nice:
        os.nice(args.nice)
    try:
        import tensorboard  # noqa: F401
        tb = os.path.join(args.out, 'tb')
    except ImportError:
        tb = None
        print('tensorboard not installed, logging to stdout only')

    if args.env == 'sim2d':
        from .sim2d import make_env
        from .scene_pool import ScenePool
        pool = ScenePool(args.pool)
        print(f'scene pool {args.pool}: {len(pool)} maps, {pool.n_scenes()} scenes not rejected', flush=True)
        env = SubprocVecEnv([make_env(args.pool, args.seed * 1000 + i, args.max_obstacles, args.shield)
                             for i in range(args.n_envs)])
        hyper = dict(n_steps=256, batch_size=2048, n_epochs=5, learning_rate=3e-4, ent_coef=args.ent_coef,
                     target_kl=args.target_kl or None)
    else:
        from .webots_env import WebotsEnv
        env = DummyVecEnv([lambda: Monitor(WebotsEnv(args.paths))])
        hyper = dict(n_steps=1024, batch_size=256, n_epochs=5, learning_rate=1e-4, ent_coef=args.ent_coef)

    policy_class = VestaSquashedPolicy if args.policy == 'squashed' else VestaPolicy
    if args.resume:
        # older runs saved the plain SB3 policy; the weights are the same, the mean gets tanh
        model = PPO.load(args.resume, env=env, device=args.device,
                         custom_objects=dict(hyper, policy_class=policy_class), tensorboard_log=tb)
        if args.rescale_action:
            with torch.no_grad():
                for row, k in enumerate(args.rescale_action):
                    model.policy.action_net.weight[row] *= k
                    model.policy.action_net.bias[row] *= k
            print(f'action rows rescaled by {args.rescale_action}')
        reset_steps = False
        if args.reset_std > 0:
            with torch.no_grad():
                model.policy.log_std.fill_(float(np.log(args.reset_std)))
            print(f'action std reset to {args.reset_std}')
    else:
        model = PPO(policy_class, env, policy_kwargs=POLICY_KWARGS, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2, seed=args.seed, device=args.device,
                    tensorboard_log=tb, verbose=1, **hyper)
        reset_steps = True
        if args.bc_steps > 0:
            bc_warm_start(model, env, args.bc_steps, args.bc_epochs, seed=args.seed)

    callbacks = [CheckpointCallback(save_freq=max(1, 1_000_000 // env.num_envs),
                                    save_path=os.path.join(args.out, 'checkpoints'), name_prefix=args.name or 'ppo'),
                 ActionStats(), ShieldStats()]
    if args.max_ram_gb > 0:
        callbacks.append(RamGuard(args.max_ram_gb))
    if args.env == 'sim2d':
        cur_path = os.path.join(args.out, 'curriculum.json')
        level = args.curriculum
        if level is None:
            level = json.load(open(cur_path))['level'] if os.path.exists(cur_path) else 0.0
        callbacks.append(Curriculum(cur_path, level))
        if args.supervisor:
            callbacks.append(Supervisor(args.out, args.pool, every=args.eval_every, workers=args.eval_workers,
                                        per_kind=args.eval_per_kind))
    try:
        model.learn(args.steps, callback=CallbackList(callbacks), reset_num_timesteps=reset_steps,
                    tb_log_name=args.name or args.env)
    finally:
        model.save(os.path.join(args.out, 'model.zip'))
        print('saved', os.path.join(args.out, 'model.zip'))
        env.close()


if __name__ == '__main__':
    main()
