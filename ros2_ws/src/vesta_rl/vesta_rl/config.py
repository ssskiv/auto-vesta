"""Constants shared by the 2D sim, the Webots env and the C++ plugin.

Anything that changes the observation or the action mapping must be changed in
vesta_neural_controller (observation.hpp / neural_controller params) as well,
and the C++ test fixture regenerated (make_obs_fixture).
"""
import math

# Vehicle (vesta.urdf / vesta_driver.py). base_link is the rear axle.
WHEELBASE = 2.648
MAX_STEER = 0.6
MAX_SPEED = 5.0
DT = 0.1  # controller period, s

# Per-tick command rate limits, applied in the sim and in the plugin.
MAX_DV = 0.3       # m/s per tick (3 m/s^2)
MAX_DDELTA = 0.1   # rad per tick (1 rad/s)

# Footprint in base_link, same as nav2_params.yaml.
FOOTPRINT_X = (-0.9, 3.7)
FOOTPRINT_Y = (-0.9, 0.9)

# Nav2 inflation layer (nav2_params.yaml) and cost values (nav2_costmap_2d).
INFLATION_RADIUS = 2.0
COST_SCALING_FACTOR = 2.0
INSCRIBED_RADIUS = 0.9
LETHAL = 254
INSCRIBED = 253
NO_INFORMATION = 255

# Lidar (lidar_joint in vesta.urdf) and local costmap obstacle range.
LIDAR_X = 0.7
LIDAR_RANGE = 15.0

# Observation: egocentric costmap patch, rows = x (forward), cols = y (left).
PATCH_SIZE = 64
PATCH_RES = 0.25
PATCH_X_MIN = -4.0
PATCH_Y_MIN = -8.0
PATCH_UNKNOWN = 0.5

# Observation: global path samples.
PATH_POINTS = 20
PATH_SPACING = 1.0
PATH_SEARCH_DIST = 10.0  # how far ahead of the last match the tracker looks

# Observation vector normalization.
PATH_SCALE = 20.0
GOAL_SCALE = 50.0
YAW_RATE_SCALE = 1.0
CTE_SCALE = 5.0

# path xy (2*PATH_POINTS) + dist_to_goal + v, yaw_rate, last_v, last_delta, cte, heading_err
VEC_SIZE = 2 * PATH_POINTS + 1 + 6

# Path speed limit (obs.speed_limit): lateral acceleration v^2 * curvature <=
# A_LAT_MAX, braking at DECEL to that speed before the curve, and reaching the
# end of the path at <= V_GOAL. The speed action is a fraction of this limit.
A_LAT_MAX = 2.0       # m/s^2
DECEL = 1.2           # m/s^2; below the car's 2.5 m/s^2 to leave room for the speed lag
LIMIT_HORIZON = 12.0  # m of path looked ahead
V_GOAL = 1.0          # m/s


def action_to_command(a_v, a_delta, v_limit):
    """Policy action in [-1, 1]^2 -> (v, delta). Forward only.

    a_v is the fraction of the path speed limit (obs.speed_limit), so the
    policy cannot drive into a curve or the goal too fast; it can only go slower.
    """
    a_v = min(1.0, max(-1.0, a_v))
    a_delta = min(1.0, max(-1.0, a_delta))
    return (a_v + 1.0) * 0.5 * v_limit, a_delta * MAX_STEER


def rate_limit(v, delta, last_v, last_delta):
    v = min(last_v + MAX_DV, max(last_v - MAX_DV, v))
    delta = min(last_delta + MAX_DDELTA, max(last_delta - MAX_DDELTA, delta))
    return v, delta


def command_to_twist(v, delta):
    """(v, delta) -> (linear.x, angular.z); vesta_driver inverts this exactly."""
    return v, v * math.tan(delta) / WHEELBASE
