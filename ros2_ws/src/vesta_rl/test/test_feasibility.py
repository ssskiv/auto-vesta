import math

import numpy as np

from vesta_rl.feasibility import MARGIN, Feasibility, dubins
from vesta_rl.obs import PathTracker
from vesta_rl.world import World

WORLD = World.from_grid(np.zeros((300, 600), dtype=bool), -10.0, -30.0, 0.2)  # 120 x 60 m, open
PATH = PathTracker(np.stack([np.arange(0, 80, 0.5), np.zeros(160)], axis=1))


def wall(occ, x, gap=None):
    """A 0.4 m thick wall across the road at x, optionally with a gap (y_lo, y_hi)."""
    for y in np.arange(-8, 8, 0.1):
        if gap and gap[0] <= y <= gap[1]:
            continue
        for dx in (0.0, 0.2):
            mx, my = WORLD.to_cell(x + dx, y)
            occ[my, mx] = True
    return occ


def route(occ):
    return Feasibility(WORLD, PATH).check(occ)


def test_dubins_reaches_the_goal_pose():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q0, q1 = rng.uniform(-20, 20, 3), rng.uniform(-20, 20, 3)
        p = dubins(q0, q1, 5.0)
        assert math.hypot(*(p[-1, :2] - q1[:2])) < 1e-6
        assert abs((p[-1, 2] - q1[2] + math.pi) % (2 * math.pi) - math.pi) < 1e-6


def test_open_road_and_wall():
    r = route(np.zeros(WORLD.shape, bool))
    assert r is not None and math.hypot(*(r[-1] - PATH.xy[-1])) < 1.5
    assert np.abs(np.hypot(*np.diff(r, axis=0).T) - 0.5).max() < 1e-6 or len(r) > 100
    assert route(wall(np.zeros(WORLD.shape, bool), 30)) is None


def test_gap_must_fit_the_car_with_margin():
    width = 1.8 + 2 * MARGIN  # 3.8 m
    assert route(wall(np.zeros(WORLD.shape, bool), 30, gap=(0.0, width + 1.0))) is not None
    assert route(wall(np.zeros(WORLD.shape, bool), 30, gap=(0.0, width - 0.6))) is None


def test_slalom_respects_turning_radius():
    """4.5 m gaps on opposite sides, 5.5 m apart sideways: an S-curve with R 5 m
    needs ~9 m, so 8 m between the walls is too tight and 30 m is fine."""
    tight = wall(wall(np.zeros(WORLD.shape, bool), 30, gap=(0.5, 5.0)), 38, gap=(-5.0, -0.5))
    loose = wall(wall(np.zeros(WORLD.shape, bool), 30, gap=(0.5, 5.0)), 60, gap=(-5.0, -0.5))
    assert route(tight) is None
    r = route(loose)
    assert r is not None


def test_route_keeps_the_margin():
    occ = np.zeros(WORLD.shape, bool)
    for x in (20, 35, 50):  # barrels on the path
        for dx in np.arange(-0.4, 0.4, 0.1):
            for dy in np.arange(-0.4, 0.4, 0.1):
                mx, my = WORLD.to_cell(x + dx, dy)
                occ[my, mx] = True
    r = route(occ)
    assert r is not None
    # base_link points: at least half the width + margin from each barrel centre (minus the barrel)
    for x in (20, 35, 50):
        assert np.hypot(r[:, 0] - x, r[:, 1]).min() > 0.9 + MARGIN
