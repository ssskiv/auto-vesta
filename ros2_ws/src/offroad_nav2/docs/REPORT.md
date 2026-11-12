# Final report: perception-to-Nav2 rebuild for the Lada Vesta (Webots)

Branch `ai` of autoware.off-road; auto-vesta changes on branch `ai` (worktree `/home/skiv/Programming/auto-vesta-ai`).
Environment: Humble Docker image `vesta-nav:latest` (decision 7.3 answered by the user), Nav2 1.1.20 (`cmd_vel` is Twist).
Controller: Regulated Pure Pursuit by default, MPPI selectable. The `ai` branch's neural controller is not used.
Nav2 tuning is intentionally unfinished (the user will complete it); see "Open issues".

## Phase table

| Phase | Result | Evidence |
|---|---|---|
| P0 environment | passed | torch 2.14.1+cu130 sees the RTX 5070 Ti; Nav2/robot_localization/stereo_image_proc/vision_msgs/collision_monitor resolve in the image |
| P1 engine + models refactor | passed | 5 pytest (ranges, identical to old wrappers, builds with network blocked, fp16 vs fp32 >= 99 %); `bench_engine`: preprocessing + 3 networks + post-processing 34 ms p50 / 36 ms p95 (GPU shared with Webots rendering: 54 ms in the node) |
| P2 geometry | passed | 10 pytest (lift < 1 mm, ground round trip < 1 cm, left = +y, bin 45 and edge flagged, horizon, RANSAC < 1 deg) |
| P3 description + TF | passed | `check_urdf` OK (xacro and Webots URDF); `tf` matches `vehicle.yaml`; optical z = forward, x = right, y = down |
| P4 sim sensors (Webots) | passed with deviations | stereo 15-20 Hz, depth 2.4-2.9 % low vs lidar from 5 to 20 m; GPS 8 Hz; IMU 41 Hz, not 100 Hz (Webots basic step 24 ms) |
| P5 localization | passed | 290 s of sim time standing: 0.000 m displacement; one `odom -> base_link` publisher (29 Hz); datum logged |
| P6 perception_node | passed in a clean run, borderline otherwise | clouds 9.6 Hz; latency p50 120 ms / p95 144 ms / max 216 ms over 382 frames with nothing else running; with tools or a lap running p95 reached 168-260 ms |
| P7 costmaps | 2 of 3 checks met | barrel lethal within 0.02 m of its lidar position (limit 0.3 m); clearing 0.3 s after the barrel is gone (limit 2 s); VRU halo NOT larger than the vehicle halo (3.29 m vs 3.31 m) because the final 2.4 m footprint inflation dominates |
| P8 vehicle interface | passed | 6 pytest (negative v -> 0, delta = atan(L w / v), clamp, rate limit, 300 ms watchdog, engage/e-stop stub) |
| P9 Nav2 in sim | partly | goals, GPS route and laps run; forward-only verified; no negative `linear.x`; referee results below; laps often stall (forward-only trap) |
| P10 detector / tracking / TL | skipped (no model); stand-in test passed | no detector or light-state model exists. BEV clustering and a nearest-centroid tracker exist. The stop-line behaviour was tested with the labelled stand-in |
| P11 elevation | skipped | no checkpoint; extrinsics convention unconfirmed |

## Measured numbers (Webots, this machine, RTX 5070 Ti laptop, networks in fp16)

- Engine bench (`python3 -m offroad_perception.bench_engine`, 1280x720 input): total 34.2 / 35.8 ms (p50 / p95); seg path 15.1 / 15.8, contour 3.5 / 3.8, ObjectSeg 15.1 / 16.0. Inside `perception_node` with Webots rendering on the same GPU: networks about 50 ms, clouds 14-20 ms.
- `perception_node` image-to-cloud latency (image stamp to cloud published, sim clock, 24 ms resolution), clean window of 40 s with nothing else running: p50 120 ms, p95 144 ms, max 216 ms (n = 382), meeting the 150 ms p95 budget. While a lap runs or tools are attached it was p50 144-168 ms and p95 168-260 ms. Of the total, stereo (SGBM on a 284-row crop) plus restamp is about 72 ms.
- Rates (40 s count, `tools` script): `/stereo/left/image_rect` 10.2 Hz (relay limited from 20 Hz on purpose: SGBM capacity ~16 Hz, a higher feed only queues), `/stereo/depth` 9.6, all `/perception/points/*` 9.6, `/perception/objects` 9.6, `/odometry/local` 29.5, `/odometry/global` 29.4, `/odometry/gps` 8.2, `/imu/data` 40.9, `/lidar` 8.2, `/cmd_vel` 19.6 Hz.
- Camera vs lidar (car stationary, `tools/eval_clouds_vs_lidar.py`): barrels at 8-12 m: median 0.23 m, p90 0.28 m, 99.6 % of camera points within 0.3 m of a lidar obstacle point; fence at 15-20 m: median 0.07 m, 93 % within 0.3 m. Barrels beyond ~20 m are not detected (depth limit). Not tested between 12 and 15 m (no object there).
- Stereo depth vs lidar (`tools/eval_depth_vs_lidar.py`): 5-10 m median -0.15 m (2.4 %), 10-15 m -0.36 m (2.9 %), 15-20 m -0.42 m (2.4 %).
- Speed: max commanded 4.0 m/s (14.4 km/h) in the default-speed runs, 3.0 m/s in the final configuration; limit 18 km/h.

## Referee runs (`/tmp/referee_results.json`, fixed barrel seeds via `/vesta/barrels_seed`, `lap_mission`, final configuration R=4.0 m, padding 0.2 m, 3 m/s)

Camera models (obstacles from the perception clouds): see `results/camera_seed*.json` and `*_bag.txt`.

| Seed | goals failed / 25 | touch penalties | reversing events | referee total | lap finished | negative `/cmd_vel` (bag) |
|---|---|---|---|---|---|---|
| 1 | 4 | 0 | 0 | 1900 | no | 0 |
| 2 | 2 | 0 | 0 | 3600 | yes | 0 |
| 3 | 15 | 0 | 0 | 1100 | no | 0 |
| 4 | 3 | 2 | 0 | 700 | no | 0 |
| 5 | 12 | 0 | 0 | 1500 | no | 0 |

Touch penalties: 2 in total over five laps, in one lap. The target of zero touches was not reached. Reversing events: zero in all runs
(the relay clamps, the planner is Dubin, the smoother and RPP forbid reverse). Laps stall because a forward-only car with a 4 m turning
radius gets trapped in the two tight corners when a barrel sits at the corner entrance; the planner then reports no valid path.

Earlier configurations are kept in `results/attempt*_*` (R=4.5 gave zero touches but 1 finished lap in 4 seeds; R=4.0 with 0.12 m padding
finished 3 of 4 laps with 1 to 3 touches per lap; padding 0.3 m pinned the car against a barrel). They show that the outcome is
sensitive to these parameters and that tuning is not done.

## Lidar-only ablation (camera models disabled)

`obstacles:=lidar`: perception is not started, the costmaps use the VLP-16 only (`results/lidar_seed*.json`). Only two seeds were run
(the user asked to stop tuning and finish), same configuration and barrel seeds as the camera runs:

| Seed | mode | goals failed / 25 | touch penalties | reversing | referee total | lap finished |
|---|---|---|---|---|---|---|
| 1 | camera | 4 | 0 | 0 | 1900 | no |
| 1 | lidar only | 3 | 0 | 0 | 3600 | yes |
| 2 | camera | 2 | 0 | 0 | 3600 | yes |
| 2 | lidar only | 1 | 0 | 0 | 3800 | yes |

In Webots the lidar is close to ground truth (360 degrees, dense, no latency issue), so the lidar-only stack is at least as good as the
camera stack here: it sees barrels beside and behind the car and to 30 m, where the camera sees a 57 degree cone to 20 m. What the learned
models add is not visible in this comparison: they add semantic classes (non-drivable ground, natural vs man-made, VRU) that do not
change the Webots result, and a camera-only path that works on a car without a lidar. A difference in the lap result between the two
modes on two seeds is within the run-to-run variation seen across attempts and should not be read as significant.

## Lidar removed from the simulation (camera-only)

At the user's request the VLP-16 was removed from the Webots world and URDF (auto-vesta `ai` commit "Remove the VLP-16 ..."); the driver
treats it as optional. There is no `/lidar` topic and no `lidar_link` frame; `obstacles:=lidar` and the `eval_*_vs_lidar` tools no longer apply.
The camera stack ran unchanged (same rates, p95 latency 168 ms in that window). GPS route (3 waypoints): reached, none missed.
Referee, seeds 1-4 (seed 5 not run; the user stopped the series), same configuration as the camera runs above
(`results/nolidar_seed*.json`):

| Seed | goals failed / 25 | touch penalties | reversing | referee total | lap finished |
|---|---|---|---|---|---|
| 1 | 14 | 0 | 0 | 900 | no |
| 2 | 0 | 1 | 0 | 3000 | yes |
| 3 | 24 | 0 | 0 | 0 | no |
| 4 | 13 | 0 | 0 | 1300 | no |

Same behaviour as before: no reversing, no negative `/cmd_vel`, the forward-only car stalls in corners; the first goals fail on seed 3 (a barrel
close to the start). Nav2 tuning is not finished.

## Traffic light

No detector and no light-state classifier exist. A stand-in (world truth, labelled as such) is wired but is not a perception result:
`tl_source:=standin` makes the driver publish the world light state, `traffic_light_node` republishes it with confidence 0 and
`track_id -1`, and `stop_wall_node` places a virtual wall 4 m before the stop line (world x 11.40 = map x 9.20) while the state is not
GREEN; `collision_monitor` stops the car at it. Test (`tools/tl_test.py`, seed 7): the goal is sent at the start of a red phase; the light cycles red 15 s, orange 3 s, green 20 s. Result:
the car moved 0.35 m and held at x = 0.55 m (bumper at 4.23 m, 5 m before the stop line at 9.2 m) for the whole red and orange phase
(about 18 s, speed 0.0, 121 wall points inside the stop polygon), and left as soon as the state turned green (bumper crossed the line 5 s
later), reaching the goal. No negative `linear.x`. The first attempt drove through the red because the wall had only ~3 points inside the
1.7 m-wide stop polygon (monitor needs 4); the wall is now dense (0.1 m spacing). The car stops earlier than necessary (wall offset 4 m
plus the monitor's 3.75-4.7 m zone); that offset is a parameter.

## Domain gap

20 frames from Webots along the lap route (`results/domain_gap/NN_rgb.png`, `NN_seg.png`, 1280x720, sim-time stamps; the `rgb`
images are the rectified left stereo camera). Visual check of the segmentation overlay (green drivable, magenta natural, yellow man-made, red
VRU, blue vehicle):

- Drivable ground: the sand track is segmented as drivable over its full width in all inspected frames; good.
- Vegetation is labelled natural; fences, walls and signs man-made. Good.
- Barrels are labelled man-made, sometimes partly drivable, never vehicle. A single barrel close ahead can be partly drivable-coloured, so the
  costmaps rely on the depth geometry (geometric cloud) for barrels, not on the labels.
- A patch over distant mountains or sky is sometimes labelled drivable (above the horizon, so it is never projected).
- The paler sand segment of the track was classified drivable in a different shade (class output is still 1).
- Not tested: pedestrians, vehicles, traffic lights (none exist in the RoboCross world except the light).
This is a mild domain gap on the track itself, not a bug in the stack. ObjectSeg checkpoint used was the snapshot copy taken at the start
(epoch 11 of 102); it keeps training.

## Deviations from the guide

1. Controller default is RPP, not MPPI: MPPI did not hold 10 Hz next to Webots plus the networks on this machine and crawled at 1-1.5 m/s. MPPI stays selectable (`controller:=mppi`) and is not tuned.
2. Planner minimum radius 4.0 m (guide: 4.5 / 5.2, user TODO). The kinematic minimum is 3.87 m.
3. Depth source: a stereo pair added to the Webots world (the user's choice), SGBM on a row crop (rows 340-624) and 10 Hz because full-frame SGBM was too slow. Depth is padded back to full frame.
4. Global costmap uses the geometric, VRU and vehicle layers only; the semantic layer (non-drivable, natural, man-made) is local only, because stale semantic marks blocked the corridor.
5. Collision monitor: a tight stop zone (half-width 0.85 m, 3.75-4.7 m ahead) plus a slow-down zone; a large stop zone froze the forward-only car.
6. The BT keeps following the last path when a replan fails (`AlwaysSuccess` fallback) instead of aborting.
7. Costmap heights are relative to the ground in the costmap frame (ground at z = -0.108 m, see below), not to a hard-coded 0.
8. `base_link` is 0.108 m above ground in Webots after settling, not 0.4 m (the 0.4 m in NN.md is the spawn drop height); GPS lever arm x = 0.4 m, not 0.3 m.
9. Guide P4 asked for IMU >= 100 Hz: Webots delivers 41 Hz (24 ms step). Calibration RMS and real-car reference bags are not applicable in the sim.
10. The perception clouds are not published in `base_footprint`; they are in the optical frame and the costmaps transform them.
11. Lap missions use `NavigateToPose` with live waypoint snapping (`lap_mission`), not plain `FollowWaypoints`, because a barrel on a centreline waypoint made the planner fail. `gps_route` (FromLL -> FollowWaypoints) is used for the GPS route test.
12. Temporal filter, class clouds and tracking are implemented; the BEV tracker is a nearest-centroid tracker, not a Kalman tracker.

## Open issues

- Nav2 tuning (planner, controller, corner traps, barrel clearance) is unfinished by agreement; MPPI performance is untested after the first attempt.
- Latency meets 150 ms p95 only in a quiet run (p95 144 ms); under load it is 170-260 ms. The GPU is shared with Webots. Candidates: smaller depth crop, a pipelined worker, TensorRT.
- VRU halo larger than the vehicle halo was asked for (guide P7) but the final inflation layer makes both ~3.3 m outer radius; the per-class inflation needs a final-layer radius smaller than the class radii, which MPPI's footprint check does not allow.
- Label bug in the training config, reported not changed: `Models/data_parsing/config/label_config/objectseg_goose.yaml:32` maps `19: 2 # traffic_light -> man-made_structures`, but class 2 is "vulnerable living" (man-made is class 1).
- The Nav2 lifecycle bring-up occasionally hangs on a lost DDS response; `restart.sh` retries and the manager is started 20 s late.
- MPPI logs an ERROR about inflation versus the circumscribed radius (3.78 m from the rear axle); harmless.

## `TODO(user)` still open

Operating domain (7.1); VLP-16 / ARS408 on the real car (7.2); detector / traffic-light model and stop option (7.5); drive-by-wire
interface (7.6, abstract stub only); measured real vehicle numbers and the planner radius (7.7); resize vs crop (7.8, resize is the
default); non-drivable cost policy lethal vs soft (7.10); checkpoints, domain gap and road approval (7.11); real venue datum lat/lon
(placeholder 54.3282, 48.3866); maintainer and license fields of all `package.xml`/`setup.py`.

## Files changed outside autoware.off-road (auto-vesta, branch `ai`, worktree `/home/skiv/Programming/auto-vesta-ai`)

- `ros2_ws/src/webots_ros2_suv/resource/vesta.urdf`: stereo links and optical frames, `base_footprint` (z -0.108), two stereo camera devices (`/stereo_raw/{left,right}`), GPS x 0.4.
- `ros2_ws/src/webots_ros2_suv/worlds/robocross.wbt`: two stereo cameras in the front slot (plus the user's offline-vendoring path rewrites copied from the `sim` working tree).
- `ros2_ws/src/webots_ros2_suv/webots_ros2_suv/vesta_driver.py`: Webots-time stamps, 16 `spin_once` per step, `VESTA_PUBLISH_ODOM_TF` switch, optional stand-in traffic-light publisher (`VESTA_STANDIN_TL`).
- `ros2_ws/src/webots_ros2_suv/protos/webots/`, `protos/*.proto`, textures, `tools/vendor_webots_assets.py`: copied from the user's uncommitted `sim` working tree (offline vendoring). The merge of `sim` into `ai` (referee, barrel generator) is commit `eb523c4`.
- Nothing was pushed. The user's `sim` working tree and `/home/skiv/autoware_data/ml_models/` were not modified (checkpoints were copied to `/home/skiv/autoware_data/ml_models_snapshot/`).

Files changed inside autoware.off-road outside the new `ROS2/` and `ros2_ws/src/offroad_nav2/docs/`: `Models/model_components/backbone.py` and
`Models/model_components/scene_seg_network.py` (`pretrained=` argument, default keeps training unchanged). No other training or
data code was touched; `freespace_seg_infer copy.py` was not deleted.
