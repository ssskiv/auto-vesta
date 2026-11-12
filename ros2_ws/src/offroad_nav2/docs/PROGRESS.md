# Nav2 + perception rebuild: progress bookmark (resume point)

STATE 2026-11-12 (end of the autonomous run): all phases done or reported; the full results, deviations and open items are in `REPORT.md`.
The user asked to stop tuning Nav2 (it will be completed later): next work = planner/controller tuning (corner traps, barrel clearance, MPPI),
then a real detector/traffic-light model (P10). The table below is the older mid-run snapshot.

Last updated: 2026-11-12. Branch `ai` of autoware.off-road; auto-vesta changes live on branch `ai`, worktree
`/home/skiv/Programming/auto-vesta-ai` (the user's `sim` working tree is untouched). Controller: MPPI (RPP as fallback).
The `ai` branch's neural controller plugin is NOT used (user instruction).

## How to run (Docker, image vesta-nav:latest, built from ros2_ws/src/offroad_nav2/docker/Dockerfile.nav)
    ros2_ws/src/offroad_nav2/docker/build.sh                      # colcon build (webots_ros2_suv + ROS2/src) into ~/.cache/vesta-nav
    NAME=nav_sim ros2_ws/src/offroad_nav2/docker/run.sh 'ros2 launch offroad_nav2 webots_nav.launch.py'
    docker exec nav_sim bash -lc 'source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; ros2 run offroad_nav2 nav_goal 60 -1 0'
Webots needs a second launch if textures look white (remote assets); the `ai` worktree has them vendored.
Checkpoint snapshots: /home/skiv/autoware_data/ml_models_snapshot (copies; the live ml_models dir is never written).

## Phase status
| Phase | State | Evidence |
|---|---|---|
| P0 environment | passed | torch cu130 sees RTX 5070 Ti; Nav2 1.1.20 (Humble, Twist) |
| P1 engine | passed | 5 pytest; bench 34 ms p50 / 36 p95 (pre+3 nets+post, 1280x720 in) |
| P2 geometry | passed | 10 pytest |
| P3 description/TF | passed | check_urdf OK; tf matches vehicle.yaml; optical z forward |
| P4 sim sensors | passed (Webots) | stereo 15 Hz, depth 2.4-2.9 % vs lidar 5-20 m; IMU 41 Hz (not 100: Webots step 24 ms) |
| P5 localization | passed | 290 s standing: 0.000 m drift; one odom->base_link publisher (29 Hz) |
| P6 perception_node | partly | clouds ~10 Hz; latency p50 ~150-170 ms, p95 ~260 ms (target 150): NOT met |
| P7 costmaps | in progress | Nav2 comes up active; costmap acceptance not yet run |
| P8 vehicle interface | unit tests pass (6) | zero-clamp relay + steering + watchdog + DBW stub |
| P9 Nav2 sim | in progress | forward-only confirmed (no negative linear.x); collision monitor tuning in progress |
| P10 detector/TL | not started | no detector exists; stand-in plan below |
| P11 elevation | skipped | no checkpoint, extrinsics convention unconfirmed |

## Findings that matter
- Webots camera and IMU plugin stamps are WALL time; `/vesta/imu/data` too. Fixed by `stamp_relay` (images) and driver sim stamps (`/imu/data`).
- vesta_driver `spin_once` handled one callback per step, so its clock lagged 96 ms: fixed (drain 16, `getTime()` stamps).
- After settling, base_link is 0.108 m above ground in Webots (0.4 is the spawn drop). base_footprint z = -0.108 (stereo + lidar agree). GPS lever x = 0.4 not 0.3.
- Setting a uint8[] message field from bytes costs ~200 ms/MB in rclpy Humble; use array.array('B', ...).
- Camera-vs-lidar (car stationary, barrels ~10 m, fence 15-20 m): 99.6 % of camera points within 0.3 m of a lidar point at 8-12 m (median 0.23), 93 % at 15-20 m (median 0.07).
- At the spawn point all barrels are 17-34 m away: beyond the 20 m depth limit.
- A large collision-monitor stop zone dead ahead freezes a forward-only car; now: stop 3.75-4.7 m, slow-down to 7.5 m.
- Stop line: world x = 11.35 (map x 9.15), line along y. Light at world (18.96, -11.73), cycles red 15 s / green 20 s (customData/state of the GenericTrafficLight robot).
- MPPI logs a harmless-looking ERROR about inflation vs circumscribed radius (footprint circumscribed radius 3.78 m from the rear axle).

## Next steps
1. Relaunch, run `nav_goal 60 -1 0` and the GPS-waypoint route; check speeds <= 18 km/h; bag `/cmd_vel`.
2. P7 acceptance: barrel lethal within 0.3 m, clears within 2 s after barrels are reseeded, VRU halo > vehicle halo.
3. Referee over 5 seeds (`/tmp/referee_results.json`), lidar-only ablation (`obstacles:=lidar`).
4. Traffic light stand-in: publish the world light state from the driver (supervisor), enable stop_wall (line pose above, wall_offset 4 m); label it STAND-IN.
5. gps waypoint follower node (no FollowGPSWaypoints in Humble), README ros2_ws/src/offroad_nav2/docs/README.md, final report.

## Open TODO(user)
Operating domain; real-car sensors; planner min radius 4.5 vs 5.2; resize vs crop; non-drivable cost policy; detector/TL model; DBW (OSCC/ros2cco); real datum lat/lon; maintainer/license fields.
