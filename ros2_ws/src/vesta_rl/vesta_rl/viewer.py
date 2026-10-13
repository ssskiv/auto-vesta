"""Watch the current policy drive in the 2D sim, next to a running training.

Runs its own Sim2DEnv with the newest checkpoint of a run and reloads it when
training writes a new one (every ~1M steps). Does not touch the training.

    ros2 run vesta_rl viewer --run runs/sim

Keys: space = pause, n = next episode, + / - = faster / slower, q = quit.
--snapshot out.png saves one frame after --snapshot-steps steps and exits.
"""
import argparse
import collections
import glob
import math
import os
import time

import numpy as np

from . import config as C
from .arbiter import SOURCE_NAMES

POOL_DEFAULT = 'src/vesta_rl/data/pool'
VIEW_HALF = 25.0  # m around the car in the world view

THEMES = {
    'dark': dict(style='dark_background', bg='#0f172a', free=(0.06, 0.09, 0.16), map=(0.62, 0.66, 0.72),
                 obstacle=(0.96, 0.62, 0.20), seen=(1.0, 0.25, 0.30), inflation=(0.35, 0.60, 1.0), inflation_alpha=0.85,
                 path='#60a5fa', trail='#4ade80', car='#f97316', car_edge='#fde68a', view='#c084fc',
                 dots='#22d3ee', text='#e5e7eb', legend='#1e293b'),
    'light': dict(style='default', bg='white', free=(1.0, 1.0, 1.0), map=(0.25, 0.25, 0.25),
                  obstacle=(0.55, 0.35, 0.15), seen=(0.85, 0.1, 0.1), inflation=(0.0, 0.0, 1.0), inflation_alpha=0.5,
                  path='tab:blue', trail='tab:green', car='tab:orange', car_edge='k', view='tab:purple',
                  dots='tab:cyan', text='black', legend='white'),
}


def newest_model(run_dir):
    files = glob.glob(os.path.join(run_dir, 'checkpoints', '*.zip')) + glob.glob(os.path.join(run_dir, 'model.zip'))
    return max(files, key=os.path.getmtime) if files else None


def load_policy(path):
    from stable_baselines3 import PPO
    return PPO.load(path, device='cpu')


class Viewer:
    def __init__(self, args):
        import matplotlib
        if args.snapshot:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon, Rectangle

        from .sim2d import Sim2DEnv

        self.args = args
        self.plt = plt
        self.theme = th = THEMES[args.theme]
        plt.style.use(th['style'])
        self.env = Sim2DEnv(args.pool, max_obstacles=args.max_obstacles, seed=args.seed)
        self.model, self.model_path, self.model_mtime = None, None, 0.0
        self.speed = args.speed
        self.paused = False
        self.skip = False
        self.stats = collections.Counter()
        self.episode = 0

        self.fig = plt.figure(figsize=(14, 7.5), facecolor=th['bg'])
        self.fig.canvas.manager.set_window_title('vesta_rl viewer')
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.6, 1])
        self.ax_w = self.fig.add_subplot(gs[0], facecolor=th['bg'])
        self.ax_p = self.fig.add_subplot(gs[1], facecolor=th['bg'])

        # world view: truth + what the lidar has marked + inflation
        self.im_world = self.ax_w.imshow(np.zeros((2, 2, 3)), origin='lower', interpolation='nearest')
        self.im_cost = self.ax_w.imshow(np.zeros((2, 2, 4)), origin='lower', interpolation='nearest')
        self.path_line, = self.ax_w.plot([], [], '-', color=th['path'], lw=1.5, label='global path')
        self.trail_line, = self.ax_w.plot([], [], '-', color=th['trail'], lw=1.5, label='driven')
        self.car_patch = Polygon(np.zeros((4, 2)), closed=True, fc=th['car'], ec=th['car_edge'], zorder=5)
        self.view_patch = Polygon(np.zeros((4, 2)), closed=True, fill=False, ec=th['view'], ls='--', lw=1)
        self.ax_w.add_patch(self.car_patch)
        self.ax_w.add_patch(self.view_patch)
        self.ax_w.set_aspect('equal')
        self.ax_w.legend(loc='upper right', fontsize=8, facecolor=th['legend'])
        self.ax_w.set_title('world (grey = static map, orange = random obstacles, red = lidar marks, blue = inflation)'
                            if args.theme == 'dark' else
                            'world (dark = static map, brown = random obstacles, red = lidar marks, blue = inflation)',
                            fontsize=9)

        # network input: the patch, forward up, left on the left
        x0, y0 = C.PATCH_X_MIN, C.PATCH_Y_MIN
        x1, y1 = x0 + C.PATCH_SIZE * C.PATCH_RES, y0 + C.PATCH_SIZE * C.PATCH_RES
        self.im_patch = self.ax_p.imshow(np.zeros((C.PATCH_SIZE, C.PATCH_SIZE)), origin='lower',
                                         extent=(y0, y1, x0, x1), cmap='magma', vmin=0, vmax=1,
                                         interpolation='nearest')
        self.ax_p.invert_xaxis()
        self.ax_p.add_patch(Rectangle((C.FOOTPRINT_Y[0], C.FOOTPRINT_X[0]), C.FOOTPRINT_Y[1] - C.FOOTPRINT_Y[0],
                                      C.FOOTPRINT_X[1] - C.FOOTPRINT_X[0], fill=False, ec=th['car'], lw=2))
        self.patch_path, = self.ax_p.plot([], [], 'o', color=th['dots'], ms=3, label='path input (20 pts)')
        self.ax_p.set_xlabel('y, m (left)')
        self.ax_p.set_ylabel('x, m (forward)')
        self.ax_p.legend(loc='lower right', fontsize=8, facecolor=th['legend'])
        self.ax_p.set_title('network input: costmap patch + path', fontsize=9)
        self.text = self.fig.text(0.01, 0.01, '', family='monospace', fontsize=9, va='bottom', color=th['text'])
        self.fig.tight_layout(rect=(0, 0.08, 1, 1))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('close_event', lambda e: setattr(self, 'closed', True))
        self.closed = False

    def on_key(self, e):
        if e.key == ' ':
            self.paused = not self.paused
        elif e.key == 'n':
            self.skip = True
        elif e.key in ('+', '='):
            self.speed = min(self.speed * 2, 64)
        elif e.key == '-':
            self.speed = max(self.speed / 2, 0.125)
        elif e.key == 'q':
            self.closed = True

    def maybe_reload(self):
        path = self.args.model or newest_model(self.args.run)
        if path is None:
            return
        mtime = os.path.getmtime(path)
        if path != self.model_path or mtime != self.model_mtime:
            try:
                self.model = load_policy(path)
            except Exception as err:  # a checkpoint still being written
                print(f'cannot load {path}: {err}')
                return
            self.model_path, self.model_mtime = path, mtime
            print(f'loaded {path}', flush=True)

    def act(self, obs):
        if self.model is None:
            return self.env.expert_action()  # nothing trained yet: show the pure-pursuit expert
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    # ---- drawing -------------------------------------------------------
    def draw(self, obs, info):
        env, car = self.env, self.env.car
        w = env.world
        cx, cy = car.x, car.y
        mx0, my0 = w.to_cell(cx - VIEW_HALF, cy - VIEW_HALF)
        mx1, my1 = w.to_cell(cx + VIEW_HALF, cy + VIEW_HALF)
        mx0, my0 = max(int(mx0), 0), max(int(my0), 0)
        mx1, my1 = min(int(mx1), w.shape[1]), min(int(my1), w.shape[0])
        sl = (slice(my0, my1), slice(mx0, mx1))
        th = self.theme
        rgb = np.empty((my1 - my0, mx1 - mx0, 3))
        rgb[:] = th['free']
        rgb[w.occupied[sl]] = th['map']
        rgb[env.occupied[sl] & ~w.occupied[sl]] = th['obstacle']
        rgb[env.costmap.seen[sl]] = th['seen']
        self.im_world.set_data(rgb)
        self.im_world.set_extent((w.ox + mx0 * w.res, w.ox + mx1 * w.res, w.oy + my0 * w.res, w.oy + my1 * w.res))

        v = env.costmap.view
        rgba = np.zeros(v.data.shape + (4,))
        rgba[..., :3] = th['inflation']
        rgba[..., 3] = np.where(v.data < C.LETHAL, v.data / 254.0 * th['inflation_alpha'], 0.0)
        self.im_cost.set_data(rgba)
        self.im_cost.set_extent((v.origin_x, v.origin_x + v.size_x * v.resolution,
                                 v.origin_y, v.origin_y + v.size_y * v.resolution))

        self.path_line.set_data(env.tracker.xy[:, 0], env.tracker.xy[:, 1])
        self.trail_line.set_data(self.trail[:, 0] if len(self.trail) else [], self.trail[:, 1] if len(self.trail) else [])
        c, s = math.cos(car.yaw), math.sin(car.yaw)
        fx = [C.FOOTPRINT_X[0], C.FOOTPRINT_X[1], C.FOOTPRINT_X[1], C.FOOTPRINT_X[0]]
        fy = [C.FOOTPRINT_Y[0], C.FOOTPRINT_Y[0], C.FOOTPRINT_Y[1], C.FOOTPRINT_Y[1]]
        self.car_patch.set_xy([(cx + c * a - s * b, cy + s * a + c * b) for a, b in zip(fx, fy)])
        px0, py0 = C.PATCH_X_MIN, C.PATCH_Y_MIN
        px1, py1 = px0 + C.PATCH_SIZE * C.PATCH_RES, py0 + C.PATCH_SIZE * C.PATCH_RES
        self.view_patch.set_xy([(cx + c * a - s * b, cy + s * a + c * b)
                                for a, b in ((px0, py0), (px1, py0), (px1, py1), (px0, py1))])
        self.ax_w.set_xlim(cx - VIEW_HALF, cx + VIEW_HALF)
        self.ax_w.set_ylim(cy - VIEW_HALF, cy + VIEW_HALF)

        self.im_patch.set_data(obs['costmap'][0])
        pts = obs['vec'][:2 * C.PATH_POINTS].reshape(-1, 2) * C.PATH_SCALE
        self.patch_path.set_data(pts[:, 1], pts[:, 0])

        total = sum(self.stats.values())
        succ = self.stats['goal'] / total if total else 0.0
        name = os.path.basename(self.model_path) if self.model_path else 'none yet: showing pure-pursuit expert'
        self.text.set_text(
            f'model: {name}    scene {env.scene[0]}/{env.scene[1]} ({env.world_kind})    episode {self.episode}  step {env.episode.steps}  x{self.speed:g}'
            f'{"  PAUSED" if self.paused else ""}\n'
            f'v {car.v:4.2f} m/s (cmd {env.last_v:4.2f})   steer {car.delta:+.2f} rad (cmd {env.last_delta:+.2f})   '
            f'return {self.ret:7.1f}   progress {env.prev_s:5.1f}/{env.tracker.length:5.1f} m   '
            f'driver {SOURCE_NAMES[env.source] if env.shield else "NN (no arbiter)"}\n'
            f'episodes: {total}  success {succ:4.0%}  {dict(self.stats)}   last end: {self.last_reason}')

    # ---- main loop -----------------------------------------------------
    def run(self):
        plt = self.plt
        self.last_reason = '-'
        if not self.args.snapshot:
            plt.show(block=False)
        while not self.closed:
            self.maybe_reload()
            obs, _ = self.env.reset()
            self.episode += 1
            self.trail, self.ret, done, info = np.empty((0, 2)), 0.0, False, {}
            while not (done or self.closed or self.skip):
                t0 = time.monotonic()
                if not self.paused:
                    for _ in range(max(1, int(self.speed))):
                        obs, r, term, trunc, info = self.env.step(self.act(obs))
                        self.ret += r
                        self.trail = np.vstack([self.trail, [[self.env.car.x, self.env.car.y]]])
                        done = term or trunc
                        if done:
                            break
                self.draw(obs, info)
                if self.args.snapshot and self.env.episode.steps >= self.args.snapshot_steps:
                    self.fig.savefig(self.args.snapshot, dpi=90)
                    print('saved', self.args.snapshot)
                    return
                if not self.args.snapshot:
                    self.fig.canvas.draw_idle()
                    period = C.DT / min(self.speed, 1.0)
                    plt.pause(max(0.001, period - (time.monotonic() - t0)))
            self.skip = False
            if done:
                self.last_reason = info.get('reason', '?')
                self.stats[self.last_reason] += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default='runs/sim', help='training --out dir; the newest checkpoint is used')
    ap.add_argument('--model', help='a specific model .zip instead of --run')
    ap.add_argument('--pool', default=POOL_DEFAULT, help='scene pool dir (scene_pool.py)')
    ap.add_argument('--max-obstacles', '--max-boxes', dest='max_obstacles', type=int, default=15)
    ap.add_argument('--speed', type=float, default=1.0, help='playback speed (1 = real time)')
    ap.add_argument('--theme', choices=sorted(THEMES), default='dark')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--snapshot', help='save one frame to this PNG and exit')
    ap.add_argument('--snapshot-steps', type=int, default=150)
    args = ap.parse_args()

    # stay light next to the training
    import torch
    torch.set_num_threads(1)
    os.nice(5)
    Viewer(args).run()


if __name__ == '__main__':
    main()
