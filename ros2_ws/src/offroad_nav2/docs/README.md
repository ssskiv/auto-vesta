# Perception to Nav2 for the Lada Vesta (Webots demo)

One ament_cmake package, `ros2_ws/src/offroad_nav2/`, containing the Python packages `offroad_perception`, `offroad_vehicle_interface`,
`offroad_bringup`, the message `TrafficLightState` (as `offroad_nav2/msg/TrafficLightState`), launch files, configs and the vehicle
description (`config/vehicle.yaml`, `urdf/`). Architecture and decisions: `INTEGRATION_GUIDE.md` in the autoware.off-road repo
(`Modules/Nav2/`). Current state and open items: `.claude/PROGRESS.md` in the repo root (local note, not tracked).
The final report is `REPORT.md`.

How to launch step by step: `LAUNCH_GUIDE.md`.

## Implementation

Runtime nodes are C++17 (`src/nodes/*.cpp`, executables with the same names as before) on a pure-C++ core (`src/core/`, `include/offroad_nav2/`).
Inference: ONNX Runtime GPU (`/opt/onnxruntime`), three fp16 graphs exported by `tools/export_onnx.py` (`make onnx`), checked against PyTorch
(agreement 99.9 % / 100 % / 99.7 % of the outputs on a real frame). Tests: GoogleTest in `test/*.cpp` (geometry, post-processing, tracking, vehicle logic)
plus `test/test_forward_only.py` for the generated Nav2 parameters. Measured Python-vs-C++ load and latency: `docs/perf/PERF_REPORT.md`.
The last Python implementation is the git tag `python-baseline`.

## Run (Docker, ROS 2 Humble)

```bash
make image                                          # the Docker image (built only if missing or outdated)
make up                                             # container + colcon build + bash; then: ros2 launch offroad_nav2 webots_nav.launch.py
ros2_ws/src/offroad_nav2/docker/build.sh                     # colcon build into ~/.cache/vesta-nav
NAME=nav_sim ros2_ws/src/offroad_nav2/docker/run.sh 'ros2 launch offroad_nav2 webots_nav.launch.py'
```

Needs: an X display, the NVIDIA container toolkit, and checkpoint copies in
`/home/skiv/autoware_data/ml_models_snapshot/` (`last_freespaceseg.pth`, `last_freespacecontour.pth`,
`last_objectseg.pth`; the live `ml_models/` directory is never written). The simulator side is the same branch (`nav2ai`) (
`/home/skiv/Programming/auto-vesta` is mounted at `/vesta`).

Launch arguments of `webots_nav.launch.py`:

| Argument | Default | Meaning |
|---|---|---|
| `gui` | `false` | Webots window |
| `tl_source` | `none` | `none` or `standin` (world truth, NOT perception) |
| `barrels_seed` | `7` | `>0` fixed barrel layout, `0` leave the world random |

Tools (inside the container): `ros2 run offroad_nav2 nav_goal X Y [YAW]`, `gps_route --map X,Y ...` or
`gps_route LAT,LON ...`, `lap_run` (one lap + referee result).

## Topics

| Topic | Type | Rate (measured) | Notes |
|---|---|---|---|
| `/stereo/left/image_rect`, `/stereo/left/camera_info` | Image, CameraInfo | ~10 Hz | sim-time stamps, `stereo_left_optical_frame` |
| `/stereo/depth` | Image 32FC1 (m) | ~10 Hz | depth-provider contract of `perception_node` |
| `/perception/points/{geometric,nondrivable,clear,vru,vehicle,manmade,natural,contour_boundary,contour_clear_end}` | PointCloud2 | ~10 Hz | image stamp, optical frame |
| `/perception/objects` | Detection3DArray | ~10 Hz | clusters of VRU/vehicle points in `odom` |
| `/perception/traffic_light` | TrafficLightState | 10 Hz | `UNKNOWN` unless `tl_source:=standin` |
| `/perception/latency_ms` | Float32 | ~10 Hz | image stamp to cloud published |
| `/odometry/local`, `/odometry/global`, `/odometry/gps` | Odometry | 29 / 29 / 8 Hz | EKFs and navsat_transform |
| `/cmd_vel_nav_raw` -> `/cmd_vel_smoothed` -> `/cmd_vel_nav` -> `/cmd_vel` | Twist | 10-20 Hz | controller, smoother, collision monitor, zero-clamp relay |

TF: `map -> odom` (ekf_global), `odom -> base_link` (ekf_local, the only publisher: the Webots driver's own broadcast is
switched off with `VESTA_PUBLISH_ODOM_TF=0`), `base_link -> base_footprint`, `base_link -> stereo_{left,right}_link ->
*_optical_frame`.

## Parameters

All configurable parameters are in YAML files under `config/` (installed with the package); launch files only reference them and apply the
launch-argument overrides (`tl_source`). Open choices are marked `TODO(user)`.

| File | Contents |
|---|---|
| `nodes.yaml` | every parameter of the C++ nodes: `stamp_relay`, `disparity_to_depth`, `perception_node`, `traffic_light_node`, `stop_wall`, `sim_sensor_adapter`, `wheel_odometry`, `zero_clamp_relay`, `cmd_to_ackermann` |
| `nav2_params.yaml` | Nav2: costmaps and layers, Smac planner, RPP controller, smoother, collision monitor, BT, lifecycle (map and BT file paths are substituted at launch) |
| `ekf.yaml`, `navsat.yaml` | localization (two EKFs, navsat_transform and its datum) |
| `stereo.yaml` | stereo matcher (SGBM) |
| `vehicle.yaml` | vehicle and camera numbers (wheelbase, footprint, stereo mount); reference for the values used in the other files |

`test/test_forward_only.py` checks that the files agree (forward-only Nav2 settings, equal stereo crop, equal GNSS datum, command chain topics).
## Known limitations

- Webots camera and IMU plugins stamp with wall time; handled by `stamp_relay` (images) and by the driver stamping with
  Webots time (`/imu/data`). Do not use `/vesta/imu/data`.
- Depth: SGBM stereo, 2.4-2.9 % low versus lidar from 5 to 20 m; trusted to 20 m; blind zone below ~2.4 m from the camera
  (nothing covers it: there is no lidar in the simulation).
- base_link is 0.108 m above ground in Webots after settling (not 0.4 m).
- The controller is Regulated Pure Pursuit only (MPPI was removed: it did not hold 10 Hz next to Webots plus the networks here).
- The models were trained on real photos; barrels are labelled inconsistently (man-made / vehicle). Geometry from depth is
  what the costmaps rely on for barrels.
- No detector, no light-state classifier, no elevation network exist yet. The traffic-light test uses a stand-in.
- Real-car drive-by-wire is an abstract logging stub (`offroad_vehicle_interface/dbw_adapter_base.py`).
