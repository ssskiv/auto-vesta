import os

import numpy as np
import pytest

MAP = os.path.join(os.path.dirname(__file__), '..', '..', 'webots_ros2_suv', 'maps', 'robocross.yaml')
PATHS = os.path.join(os.path.dirname(__file__), '..', 'data', 'paths.npz')
pytestmark = pytest.mark.skipif(not (os.path.exists(MAP) and os.path.exists(PATHS)),
                                reason='needs maps/robocross.yaml and data/paths.npz')


@pytest.fixture(scope='module')
def pool(tmp_path_factory):
    """A small pool: one map of every kind, 4 layouts each."""
    import argparse
    from vesta_rl import scene_pool
    out = str(tmp_path_factory.mktemp('pool'))
    scene_pool.generate(argparse.Namespace(
        out=out, map=MAP, paths=PATHS, maps=len(scene_pool.KINDS), layouts=4, max_obstacles=15,
        robocross_share=None, sheets=1, workers=2, nice=0, seed=0, overwrite=False))
    return out


def test_pool_has_every_kind_and_previews(pool):
    from vesta_rl import scene_pool
    p = scene_pool.ScenePool(pool)
    assert sorted(p.kind(i) for i in range(len(p))) == sorted(k[0] for k in scene_pool.KINDS)
    assert p.n_scenes() == len(p) * 4
    assert os.path.exists(os.path.join(pool, 'previews', 'm000_p0.png'))
    img = scene_pool.render(p, 0, 0, 200)
    assert img.shape == (200, 200, 3)


def test_rejected_scenes_are_never_used(pool):
    from vesta_rl import scene_pool
    from vesta_rl.sim2d import Sim2DEnv
    scene_pool.write_rejected(pool, {0, 1, 2}, {3: {0, 1, 2}})
    try:
        env = Sim2DEnv(pool, seed=0)
        seen = set()
        for _ in range(40):
            env.reset()
            seen.add(env.scene)
        assert all(i >= 3 for i, _ in seen) and (3, 0) not in seen and (3, 3) in seen
    finally:
        scene_pool.write_rejected(pool, set(), {})


def test_expert_reaches_goal_without_obstacles(pool):
    from vesta_rl.sim2d import Sim2DEnv
    env = Sim2DEnv(pool, seed=0, max_obstacles=0, shield=False)  # track geometry only
    for i in range(len(env.pool)):
        env.reset(options={'scene': (i, 0)})
        done = False
        while not done:
            _, _, term, trunc, info = env.step(env.expert_action())
            done = term or trunc
        assert info['reason'] == 'goal', (i, info)


def test_spaces(pool):
    from vesta_rl.sim2d import Sim2DEnv
    env = Sim2DEnv(pool, seed=0)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    obs, r, *_ = env.step(np.zeros(2, dtype=np.float32))
    assert env.observation_space.contains(obs) and np.isfinite(r)


def test_procedural_tracks_are_drivable():
    from vesta_rl import procgen
    from vesta_rl.paths import max_curvature
    from vesta_rl.world import distance_to_true
    rng = np.random.default_rng(0)
    for layout in procgen.LAYOUTS:
        for _ in range(3):
            world, path = procgen.generate(rng, layout=layout)
            assert max_curvature(path) < 1.0 / 3.9
            mx, my = world.to_cell(path[:, 0], path[:, 1])
            assert (distance_to_true(world.occupied)[my, mx] * world.res).min() >= procgen.PATH_CLEARANCE - world.res
            assert not procgen.self_close(path)


def test_reachability_check():
    """A wall across the corridor blocks the way; a 4.5 m gap in it passes the quick
    check (car 1.8 m + 1 m margin on each side), 3 m does not."""
    import math
    from vesta_rl import procgen
    from vesta_rl.obs import PathTracker
    from vesta_rl.scene_pool import reachable
    world, path = procgen.generate(np.random.default_rng(0), layout='open')
    tr = PathTracker(path)
    assert reachable(world, world.occupied, tr)
    s = tr.length / 2
    x, y = tr.point_at(s)
    nx, ny = tr.point_at(s + 0.5)
    h = math.atan2(ny - y, nx - x)

    def wall(gap):
        occ = world.occupied.copy()
        for off in np.arange(-8.0, 8.0, world.res / 2):
            if abs(off) < gap / 2:
                continue
            for t in (-0.1, 0.0, 0.1):  # 0.3 m thick
                mx, my = world.to_cell(x - math.sin(h) * off + math.cos(h) * t,
                                       y + math.cos(h) * off + math.sin(h) * t)
                occ[my, mx] = True
        return occ

    assert not reachable(world, wall(0.0), tr)
    assert reachable(world, wall(4.5), tr)
    assert not reachable(world, wall(3.0), tr)


def test_every_scene_has_a_route_clear_of_obstacles(pool):
    from vesta_rl import feasibility as F
    from vesta_rl.scene_pool import ScenePool, blocked_grid
    from vesta_rl.world import distance_to_true
    p = ScenePool(pool)
    for i in range(len(p)):
        world, path = p.world(i), p.path(i)
        for j in range(p.n_layouts(i)):
            route = p.route(i, j)
            assert np.hypot(*(route[0] - path[0])) < 1e-3 and np.hypot(*(route[-1] - path[-1])) <= F.GOAL_TOL
            clear = distance_to_true(blocked_grid(world, *p.layout(i, j))) * world.res
            mx, my = world.to_cell(route[:, 0], route[:, 1])
            ok = world.inside(mx, my)
            assert clear[my[ok], mx[ok]].min() >= C_HALF_WIDTH + F.MARGIN - 0.2, (i, j)


C_HALF_WIDTH = 0.9


def test_expert_drives_around_the_obstacles(pool):
    """The BC expert follows the planner route: no collisions on any scene."""
    from vesta_rl.sim2d import Sim2DEnv
    env = Sim2DEnv(pool, randomize=False, shield=False)
    env.P_NO_OBSTACLES = 0.0
    reasons = []
    for i in range(len(env.pool)):
        for j in range(env.pool.n_layouts(i)):
            env.reset(options={'scene': (i, j)})
            done = False
            while not done:
                _, _, term, trunc, info = env.step(env.expert_action())
                done = term or trunc
            reasons.append(info['reason'])
    assert reasons.count('goal') == len(reasons), reasons


def test_arbiter_stops_before_a_wall_and_lets_the_policy_drive_on_open_road():
    import math
    from vesta_rl import arbiter
    from vesta_rl.obs import CostmapView, PathTracker
    tr = PathTracker(np.stack([np.arange(0, 60, 0.5), np.zeros(120)], axis=1))
    data = np.zeros((200, 700), dtype=np.uint8)  # 0.1 m cells, x in [-5, 65), y in [-10, 10)
    cm = CostmapView(data, -5.0, -10.0, 0.1)
    go = (3.0, 0.0)
    cmd, src = arbiter.arbitrate(cm, tr, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, go, 5.0)
    assert src == arbiter.NN and cmd == go
    data[:, int((8.0 + 5.0) / 0.1):int((8.4 + 5.0) / 0.1)] = 254  # wall across the road 8 m ahead
    cmd, src = arbiter.arbitrate(cm, tr, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, go, 5.0)
    assert src == arbiter.STOP and cmd[0] == 0.0


def test_override_penalty_prefers_proposals_close_to_what_was_sent():
    from vesta_rl.reward import K_SHIELD, override_penalty
    stop = (0.0, 0.1)
    assert override_penalty((0.0, 0.1), stop) == K_SHIELD
    assert override_penalty((1.0, 0.1), stop) < override_penalty((4.0, 0.1), stop)
    assert override_penalty((1.0, 0.1), stop) < override_penalty((1.0, -0.5), stop)


def test_kind_weights_steer_sampling(pool):
    from vesta_rl.scene_pool import ScenePool
    p = ScenePool(pool)
    rng = np.random.default_rng(0)
    p.set_kind_weights({'open': 1.0})
    assert {p.kind(p.sample(rng)[0]) for _ in range(50)} == {'open'}
    p.set_kind_weights(None)
    assert len({p.kind(p.sample(rng)[0]) for _ in range(200)}) == len(p.kind_maps)
