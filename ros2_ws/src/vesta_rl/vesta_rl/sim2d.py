"""Fast 2D gym env for bulk RL: bicycle model + reviewed scene pool + simulated local costmap."""
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import arbiter
from . import config as C
from .obs import PathTracker, build_patch, build_vec, footprint_world, wrap_angle
from .reward import Episode, override_penalty, speed_limit, step_reward
from .scene_pool import ScenePool, ghost_cells, shape_cells
from .world import LocalCostmap


class VehicleParams:
    """First-order speed/steer response. Defaults = calibrate.py nominal."""

    def __init__(self, tau_v=0.6, accel_max=2.5, tau_delta=0.1, steer_rate=1.0):
        self.tau_v = tau_v
        self.accel_max = accel_max
        self.tau_delta = tau_delta
        self.steer_rate = steer_rate

    @classmethod
    def random(cls, rng):
        return cls(tau_v=rng.uniform(0.3, 1.2), accel_max=rng.uniform(1.5, 3.5),
                   tau_delta=rng.uniform(0.05, 0.25), steer_rate=rng.uniform(0.5, 1.5))


class Vehicle:
    SUBSTEPS = 5

    def __init__(self, params, x=0.0, y=0.0, yaw=0.0):
        self.p = params
        self.x, self.y, self.yaw = x, y, yaw
        self.v = 0.0
        self.delta = 0.0
        self.yaw_rate = 0.0

    def step(self, v_cmd, delta_cmd, dt=C.DT):
        h = dt / self.SUBSTEPS
        yaw0 = self.yaw
        for _ in range(self.SUBSTEPS):
            dv = (v_cmd - self.v) * min(1.0, h / self.p.tau_v)
            self.v += max(-self.p.accel_max * h, min(self.p.accel_max * h, dv))
            dd = (delta_cmd - self.delta) * min(1.0, h / self.p.tau_delta)
            self.delta += max(-self.p.steer_rate * h, min(self.p.steer_rate * h, dd))
            self.x += self.v * math.cos(self.yaw) * h
            self.y += self.v * math.sin(self.yaw) * h
            self.yaw = wrap_angle(self.yaw + self.v * math.tan(self.delta) / C.WHEELBASE * h)
        self.yaw_rate = wrap_angle(self.yaw - yaw0) / dt


def observation_space():
    return spaces.Dict({
        'costmap': spaces.Box(0.0, 1.0, (1, C.PATCH_SIZE, C.PATCH_SIZE), np.float32),
        'vec': spaces.Box(-np.inf, np.inf, (C.VEC_SIZE,), np.float32),
    })


class Sim2DEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, pool, randomize=True, max_speed=C.MAX_SPEED, max_obstacles=15, shield=True, seed=None):
        self.pool = pool if isinstance(pool, ScenePool) else ScenePool(pool)
        self.shield = shield          # run the plugin's arbiter (NN / RPP / STOP) in the loop
        self.source = arbiter.NN
        self.randomize = randomize
        self.max_speed = max_speed
        self.max_obstacles = max_obstacles
        self.full_obstacles = max_obstacles
        self.observation_space = observation_space()
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.rng = np.random.default_rng(seed)

    # ---- episode setup -------------------------------------------------
    # Scenes come from a pool generated and reviewed before training (scene_pool.py).
    # Its obstacles are unknown to the global planner: the path is planned on the
    # static map only, like Nav2's global costmap (static layer only).
    P_NO_OBSTACLES = 0.15

    # curriculum: 0 = easy (at most MIN_OBSTACLES of a layout's obstacles) .. 1 = all of them
    MIN_OBSTACLES = 3

    def set_kind_weights(self, weights):
        """Sampling weight per map kind (supervisor: more of what the policy fails)."""
        self.pool.set_kind_weights(weights)

    def set_difficulty(self, level):
        level = min(1.0, max(0.0, level))
        self.max_obstacles = int(round(self.MIN_OBSTACLES + level * (self.full_obstacles - self.MIN_OBSTACLES)))
        return self.max_obstacles

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = self.rng
        rnd = self.randomize
        i, j = self.scene = options['scene'] if options and 'scene' in options else self.pool.sample(r)
        self.world, self.world_kind = self.pool.world(i), self.pool.kind(i)
        self.tracker = PathTracker(self.pool.path(i))
        self.route = PathTracker(self.pool.route(i, j))  # the generator's planner route, for the expert
        obstacles, ghosts = self.pool.layout(i, j)
        if r.random() < self.P_NO_OBSTACLES:
            obstacles = obstacles[:0]
        elif len(obstacles) > self.max_obstacles:  # a subset of a drivable layout is drivable
            obstacles = obstacles[np.sort(r.choice(len(obstacles), self.max_obstacles, replace=False))]
        occupied = self.world.occupied.copy()
        for row in obstacles:
            mx, my = shape_cells(self.world, int(row[0]), *row[1:])
            occupied[my, mx] = True
        path = self.tracker.xy

        self.costmap = LocalCostmap(self.world, occupied, r, drop_prob=r.uniform(0.0, 0.15) if rnd else 0.0)
        self.occupied = occupied
        if rnd and len(ghosts):
            self.costmap.add_ghost_cells(*ghost_cells(self.world, ghosts))
        self.obs_noise = (0.05, 0.02) if rnd else (0.0, 0.0)

        x0, y0 = path[0]
        x1, y1 = path[min(2, len(path) - 1)]
        yaw = math.atan2(y1 - y0, x1 - x0)
        if rnd:
            lat = r.uniform(-0.3, 0.3)
            x0, y0 = x0 - math.sin(yaw) * lat, y0 + math.cos(yaw) * lat
            yaw += r.uniform(-0.2, 0.2)
        params = VehicleParams.random(r) if rnd else VehicleParams()
        self.car = Vehicle(params, x0, y0, yaw)
        self.last_v, self.last_delta = 0.0, 0.0
        max_steps = int((self.tracker.length / 1.5 + 30.0) / C.DT)
        self.episode = Episode(self.tracker.length, max_steps)

        self.costmap.scan(self.car.x, self.car.y, self.car.yaw)
        obs, (s_proj, _, _) = self._obs()
        self.prev_s = s_proj
        return obs, {}

    # ---- stepping ------------------------------------------------------
    def _obs(self):
        c = self.car
        v = c.v + self.rng.normal(0.0, self.obs_noise[0]) if self.obs_noise[0] else c.v
        w = c.yaw_rate + self.rng.normal(0.0, self.obs_noise[1]) if self.obs_noise[1] else c.yaw_rate
        vec, proj = build_vec(self.tracker, c.x, c.y, c.yaw, v, w, self.last_v, self.last_delta, self.max_speed)
        patch = build_patch(self.costmap.view, c.x, c.y, c.yaw)
        return {'costmap': patch[None], 'vec': vec}, proj

    def _collided(self):
        fx, fy = footprint_world(self.car.x, self.car.y, self.car.yaw)
        mx, my = self.world.to_cell(fx, fy)
        ok = self.world.inside(mx, my)
        return bool(self.occupied[my[ok], mx[ok]].any())

    def _footprint_cost(self):
        fx, fy = footprint_world(self.car.x, self.car.y, self.car.yaw)
        cost = self.costmap.view.cost_at(fx, fy).astype(np.float32)
        cost[cost == C.NO_INFORMATION] = 0.0
        return float(cost.mean() / C.LETHAL)

    def step(self, action):
        # prev_s is the projection the policy's observation was built from
        v_lim = speed_limit(self.tracker, self.prev_s, self.max_speed)
        v_cmd, d_cmd = C.action_to_command(float(action[0]), float(action[1]), v_lim)
        v_cmd, d_cmd = C.rate_limit(v_cmd, d_cmd, self.last_v, self.last_delta)
        proposed = (v_cmd, d_cmd)
        if self.shield:  # what Nav2 would actually send: the policy, RPP or a stop
            c = self.car
            (v_cmd, d_cmd), self.source = arbiter.arbitrate(
                self.costmap.view, self.tracker, c.x, c.y, c.yaw, c.v, self.last_v, self.last_delta,
                (v_cmd, d_cmd), v_lim)
        dv, dd = v_cmd - self.last_v, d_cmd - self.last_delta
        self.last_v, self.last_delta = v_cmd, d_cmd
        self.car.step(v_cmd, d_cmd)
        self.costmap.scan(self.car.x, self.car.y, self.car.yaw)

        obs, (s_proj, cte, heading_err) = self._obs()
        overspeed = max(0.0, self.car.v - speed_limit(self.tracker, s_proj, self.max_speed))
        reward = step_reward(s_proj - self.prev_s, cte, heading_err, self._footprint_cost(), dv, dd,
                             self.max_speed, overspeed)
        if self.source != arbiter.NN:
            reward -= override_penalty(proposed, (v_cmd, d_cmd), self.max_speed)
        self.prev_s = s_proj
        ex, ey = self.tracker.xy[-1]
        dist_end = math.hypot(self.car.x - ex, self.car.y - ey)
        r_end, terminated, truncated, success = self.episode.check(s_proj, dist_end, cte, self._collided())
        info = {'source': self.source}
        if terminated or truncated:
            info.update(is_success=success, reason=self.episode.reason, kind=self.world_kind)
        return obs, reward + r_end, terminated, truncated, info

    # ---- expert for the BC warm start ---------------------------------
    def expert_action(self, cruise=3.0):
        """Pure pursuit on the scene's planner route, which goes around the
        obstacles (the generator checked it with the turning radius and margin)."""
        c = self.car
        _, _, s_route, _ = self.route.project(c.x, c.y)  # only the expert moves this tracker
        _, _, s_proj, _ = self.tracker.peek(c.x, c.y)
        ld = max(3.0, 1.2 * c.v)
        tx, ty = self.route.point_at(s_route + ld)
        dx, dy = tx - c.x, ty - c.y
        by = -math.sin(c.yaw) * dx + math.cos(c.yaw) * dy
        delta = max(-C.MAX_STEER, min(C.MAX_STEER, math.atan2(2.0 * C.WHEELBASE * by, ld * ld)))
        v = min(cruise * (1.0 - min(abs(delta) / C.MAX_STEER, 0.5)), speed_limit(self.route, s_route, self.max_speed))
        if self.tracker.length - s_proj < 5.0:
            v = min(v, 1.0)
        v_lim = speed_limit(self.tracker, s_proj, self.max_speed)
        a_v = min(1.0, 2.0 * v / v_lim - 1.0) if v_lim > 1e-6 else -1.0
        return np.array([a_v, delta / C.MAX_STEER], dtype=np.float32)


def make_env(pool_dir, seed, max_obstacles=15, shield=True):
    """Env factory for SubprocVecEnv (picklable, no reference to train.py)."""
    def _init():
        from stable_baselines3.common.monitor import Monitor
        return Monitor(Sim2DEnv(pool_dir, seed=seed, max_obstacles=max_obstacles, shield=shield))
    return _init
