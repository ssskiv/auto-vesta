import math

import numpy as np

from vesta_rl import config as C
from vesta_rl.obs import CostmapView, PathTracker, build_patch, build_vec


def test_patch_orientation():
    """An obstacle 5 m ahead and 2 m left lands in the right patch cell."""
    data = np.zeros((400, 400), dtype=np.uint8)  # 40 m: the whole patch is inside
    cm = CostmapView(data, -20.0, -20.0, 0.1)
    yaw = math.pi / 2  # facing +y: 5 m ahead, 2 m left = world (-2, 5)
    mx, my = int((-2.0 + 20.0) / 0.1), int((5.0 + 20.0) / 0.1)
    data[my - 3:my + 3, mx - 3:mx + 3] = C.LETHAL  # 0.6 m block (patch samples every 0.25 m)
    patch = build_patch(cm, 0.0, 0.0, yaw)
    i = int((5.0 - C.PATCH_X_MIN) / C.PATCH_RES)
    j = int((2.0 - C.PATCH_Y_MIN) / C.PATCH_RES)
    assert patch[i - 1:i + 2, j - 1:j + 2].max() == 1.0
    assert patch.sum() == patch[i - 2:i + 3, j - 2:j + 3].sum()  # nothing elsewhere


def test_patch_unknown_outside_costmap():
    cm = CostmapView(np.zeros((10, 10), dtype=np.uint8), 0.0, 0.0, 0.1)
    patch = build_patch(cm, 100.0, 100.0, 0.0)
    assert np.all(patch == C.PATCH_UNKNOWN)


def test_vec_straight_path():
    path = np.stack([np.arange(0, 50, 0.5), np.zeros(100)], axis=1)
    tracker = PathTracker(path)
    vec, (s, cte, he) = build_vec(tracker, 10.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0)
    assert math.isclose(s, 10.0)
    assert math.isclose(cte, -1.0)        # path is 1 m to the right
    assert abs(he) < 1e-9
    assert math.isclose(vec[0] * C.PATH_SCALE, 1.0, abs_tol=1e-5)    # first point 1 m ahead
    assert math.isclose(vec[1] * C.PATH_SCALE, -1.0, abs_tol=1e-5)


def test_tracker_does_not_jump_to_far_loop_part():
    # out 30 m and back 2 m to the side: the robot at the start must match the start
    out = np.stack([np.arange(0, 30, 0.5), np.zeros(60)], axis=1)
    back = np.stack([np.arange(30, 0, -0.5), np.full(60, 2.0)], axis=1)
    tracker = PathTracker(np.concatenate([out, back]))
    _, _, s, _ = tracker.project(1.0, 1.9)
    assert s < 5.0


def test_twist_roundtrip_matches_driver():
    for v, d in [(3.0, 0.3), (1.0, -0.5), (5.0, 0.0)]:
        lin, ang = C.command_to_twist(v, d)
        # vesta_driver: webots_steer = atan(-ang * L / lin) = -delta
        assert math.isclose(math.atan(-ang * C.WHEELBASE / lin), -d, abs_tol=1e-12)


def test_curve_speed_limit():
    from vesta_rl.reward import A_LAT_MAX, speed_limit
    straight = PathTracker(np.stack([np.arange(0, 50, 0.5), np.zeros(100)], axis=1))
    assert speed_limit(straight, 0.0) == C.MAX_SPEED
    r = 4.5  # tightest turn in the procedural tracks
    th = np.arange(0, math.pi, 0.5 / r)
    circle = PathTracker(np.stack([r * np.sin(th), r - r * np.cos(th)], axis=1))
    assert abs(speed_limit(circle, 1.0) - math.sqrt(A_LAT_MAX * r)) < 0.3   # ~3 m/s in the turn
    # 3 m before the turn: braking envelope sqrt(v_turn^2 + 2 * DECEL * 3)
    from vesta_rl.reward import DECEL
    approach = np.stack([np.arange(-12, 0, 0.5), np.zeros(24)], axis=1)
    both = PathTracker(np.concatenate([approach, np.stack([r * np.sin(th), r - r * np.cos(th)], axis=1)]))
    profile = [speed_limit(both, s) for s in np.arange(0.0, 14.0, 1.0)]
    assert profile[0] == C.MAX_SPEED                          # 12 m before: room to brake from 5 m/s
    assert all(b <= a + 1e-9 for a, b in zip(profile, profile[1:]))  # slows down smoothly ...
    assert abs(profile[-1] - math.sqrt(A_LAT_MAX * r)) < 0.3  # ... to the turn speed
    # braking envelope: never asks for more than DECEL (1 m sampling slack)
    for d, v in enumerate(profile):
        assert v <= math.sqrt(A_LAT_MAX * r + 2 * DECEL * max(0.0, 12.0 - d + 1.0)) + 1e-6


def test_speed_limit_brakes_to_goal():
    from vesta_rl.reward import DECEL, V_GOAL, speed_limit
    straight = PathTracker(np.stack([np.arange(0, 50, 0.5), np.zeros(100)], axis=1))
    end = straight.length
    assert abs(speed_limit(straight, end) - V_GOAL) < 1e-9
    assert abs(speed_limit(straight, end - 2.0) - math.sqrt(V_GOAL ** 2 + 2 * DECEL * 2.0)) < 1e-9
    assert speed_limit(straight, end - 10.0) == C.MAX_SPEED
