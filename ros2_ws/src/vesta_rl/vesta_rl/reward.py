"""Reward and episode end conditions shared by the 2D sim and the Webots env."""
import math

from . import config as C

K_PROGRESS = 1.0     # per metre along the path
K_CTE = 0.05         # per metre of cross-track error
K_HEADING = 0.05     # per radian of heading error
K_COST = 0.5         # mean footprint cost in [0, 1]
K_JERK = 0.05        # normalized command change
K_TIME = 0.01        # per step
K_OVERSPEED = 0.3    # per m/s above the curve speed limit, per step
K_SHIELD = 0.1       # per step the arbiter overrides the policy (RPP fallback or STOP)
K_OVERRIDE = 0.3     # x how far the policy's command was from what the arbiter sent instead

# speed_limit lives in obs.py (it is part of the action contract now)
from .config import A_LAT_MAX, DECEL, LIMIT_HORIZON, V_GOAL  # noqa: E402,F401
from .obs import speed_limit  # noqa: E402,F401

R_COLLISION = -75.0
R_GOAL = 80.0
R_STUCK = -50.0      # same as a collision: stopping in front of an obstacle must not be a safe way out
R_OFF_PATH = -20.0

CTE_LIMIT = 10.0      # m from the path ends the episode
GOAL_TOL = 1.5       # m from the path end
STUCK_STEPS = 100    # 10 s ...
STUCK_DIST = 1.0     # ... with less progress than this ends the episode


def step_reward(ds, cte, heading_err, footprint_cost, dv, ddelta, max_speed=C.MAX_SPEED, overspeed=0.0):
    """overspeed: max(0, v - speed_limit(...))."""
    ds = min(1.0, max(-1.0, ds))  # guards against tracker jumps
    return (K_PROGRESS * ds
            - K_CTE * abs(cte)
            - K_HEADING * abs(heading_err)
            - K_COST * footprint_cost
            - K_JERK * (abs(dv) / max_speed + abs(ddelta) / C.MAX_STEER)
            - K_OVERSPEED * overspeed
            - K_TIME)


def override_penalty(proposed, sent, max_speed=C.MAX_SPEED):
    """When the arbiter overrides the policy, the policy's own action has no effect,
    so PPO gets no signal on what it should have done. This gives one: the closer
    the proposal was to the command actually sent (RPP's, or the stop), the smaller
    the penalty. E.g. while STOP holds the car, asking for less speed is better."""
    return K_SHIELD + K_OVERRIDE * (abs(proposed[0] - sent[0]) / max_speed
                                    + abs(proposed[1] - sent[1]) / C.MAX_STEER)


class Episode:
    """Tracks progress and decides when an episode ends."""

    def __init__(self, path_length, max_steps):
        self.path_length = path_length
        self.max_steps = max_steps
        self.steps = 0
        self.s_hist = []

    def check(self, s_proj, dist_to_end, cte, collided):
        """Returns (terminal_reward, terminated, truncated, success); sets self.reason."""
        self.steps += 1
        self.s_hist.append(s_proj)
        self.reason = None
        if collided:
            self.reason = 'collision'
            return R_COLLISION, True, False, False
        if dist_to_end < GOAL_TOL and self.path_length - s_proj < GOAL_TOL:
            self.reason = 'goal'
            return R_GOAL, True, False, True
        if abs(cte) > CTE_LIMIT:
            self.reason = 'off_path'
            return R_OFF_PATH, True, False, False
        if len(self.s_hist) > STUCK_STEPS and s_proj - self.s_hist[-STUCK_STEPS - 1] < STUCK_DIST:
            self.reason = 'stuck'
            return R_STUCK, True, False, False
        if self.steps >= self.max_steps:
            self.reason = 'timeout'
            return 0.0, False, True, False
        return 0.0, False, False, False
