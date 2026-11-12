# How to launch the Vesta perception + Nav2 demo (Webots, no lidar)

Everything runs in Docker (ROS 2 Humble). Nothing is installed on the host. The simulated car has **no lidar**: obstacles come only
from the stereo camera and the three networks.

## 1. Prerequisites (once)

| Needed | Check |
|---|---|
| Docker with the NVIDIA container toolkit | `docker run --rm --gpus all ubuntu nvidia-smi` |
| An X display on the host (Webots camera rendering needs OpenGL, even with `gui:=false`) | `echo $DISPLAY` (`:0`) |
| Base image `vesta-sim:latest` (the auto-vesta image) | `docker images vesta-sim` |
| auto-vesta branch `nav2ai` checked out as a worktree at `/home/skiv/Programming/auto-vesta-ai` (this package lives in it: `ros2_ws/src/offroad_nav2/`) | `git -C /home/skiv/Programming/auto-vesta worktree list` |
| The `autoware.off-road` repo (only its `Models/` directory is used, mounted read-only as `/offroad`) | `ls /home/skiv/Programming/autoware.off-road/Models` |
| Checkpoint copies in `/home/skiv/autoware_data/ml_models_snapshot/` (`last_freespaceseg.pth`, `last_freespacecontour.pth`, `last_objectseg.pth`) | `ls /home/skiv/autoware_data/ml_models_snapshot` |

If the worktree is missing: `cd /home/skiv/Programming/auto-vesta && git worktree add ../auto-vesta-ai nav2ai`.
To refresh the ObjectSeg snapshot while training runs, copy the file; never point the demo at the live `ml_models/` folder:
`cp /home/skiv/autoware_data/ml_models/ObjectSeg/last_objectseg.pth /home/skiv/autoware_data/ml_models_snapshot/`.

All commands below are run from `/home/skiv/Programming/auto-vesta-ai` (the `nav2ai` worktree). The paths inside the scripts are absolute, so the
repositories must stay where they are.

## 2. Build (once, and after changing code)

```bash
# 1. image with PyTorch/CUDA, stereo_image_proc, Nav2 packages (about 10 min the first time)
docker build -f ros2_ws/src/offroad_nav2/docker/Dockerfile.nav -t vesta-nav:latest ros2_ws/src/offroad_nav2/docker

# 2. colcon build of the Webots package and offroad_nav2 (uses --symlink-install, build dir ~/.cache/vesta-nav)
ros2_ws/src/offroad_nav2/docker/build.sh
```

Python edits take effect after a restart of the launch; run `build.sh` again only after adding a package, an entry point, a launch
file, or a message.

## 3. Launch the whole demo

```bash
NAME=nav_sim ros2_ws/src/offroad_nav2/docker/run.sh 'ros2 launch offroad_nav2 webots_nav.launch.py'
```

It starts Webots, the stereo depth chain, localization (two EKFs and navsat), the perception node (three networks on the GPU),
Nav2 (planner, controller, smoother, collision monitor, zero-clamp relay), and sets fixed barrel layout seed 7 after 20 s.
Ready after about 60 to 90 s: the log shows Nav2 `Managed nodes are active`.

Or let the helper do build, start and wait until Nav2 is active (recommended: it retries the occasional Nav2 bring-up hang):

```bash
ros2_ws/src/offroad_nav2/docker/restart.sh                       # defaults
ros2_ws/src/offroad_nav2/docker/restart.sh barrels_seed:=3 gui:=true
```

### Launch arguments

| Argument | Default | Meaning |
|---|---|---|
| `gui` | `false` | `true` opens the Webots window (the script runs `xhost +local:` for you) |
| `models` | `true` | `false` = no networks, depth-only geometric obstacles |
| `controller` | `rpp` | `rpp` Regulated Pure Pursuit, `mppi` (heavy, untuned) |
| `barrels_seed` | `7` | `>0` fixed barrel layout, `0` leave the world random |
| `tl_source` | `none` | `standin` = traffic light state from the world (a labelled stand-in, not perception) and a stop wall at the stop line |
| `obstacles` | `camera` | `lidar` is **not usable**: the lidar was removed from the simulation |

Example with the traffic-light stand-in and the window:
`ros2_ws/src/offroad_nav2/docker/restart.sh gui:=true tl_source:=standin barrels_seed:=5`

## 4. Drive it (second terminal, same container)

The demo runs in a container named `nav_sim`. Open a shell in it:

```bash
docker exec -it nav_sim bash -lc 'source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; bash'
```

Inside (or as a one-liner with `docker exec nav_sim bash -lc '... command ...'`):

```bash
# one goal in the map frame: x y [yaw]
ros2 run offroad_nav2 nav_goal 60 -1 0

# GPS waypoints (map coordinates converted to lat/lon with the sim datum, then lat/lon -> map via /fromLL -> FollowWaypoints)
ros2 run offroad_nav2 gps_route --map 30,-2 60,-1.5 88,-2
ros2 run offroad_nav2 gps_route 54.328182,48.387062 54.328187,48.387524      # real lat,lon list

# one lap of the RoboCross track + the referee result (adjusts waypoints to free space live)
ros2 run offroad_nav2 lap_mission --timeout 900 --out /tmp/lap.json
cat /tmp/referee_results.json                      # score, touches, reversing events

# move the car to a map pose (x y yaw), e.g. back to the start
python3 /vesta/ros2_ws/src/offroad_nav2/tools/teleport.py 0.2 -0.7 0

# new barrel layout without restarting
ros2 topic pub --once /vesta/barrels_seed std_msgs/msg/Int32 "{data: 4}"
```

Map frame: origin is the start position; world = map + (2.2, -1.6). The RoboCross start is at map (0.2, -0.7) facing +x.
The start-line stop line is at map x = 9.2. Limits: forward-only, max 3.0 m/s (10.8 km/h) in the current configuration.

### Several referee runs in a row

```bash
TAG=nolidar ros2_ws/src/offroad_nav2/docker/run_seeds.sh "1 2 3 4 5"      # restarts the demo per seed, records a /cmd_vel bag
ls ros2_ws/src/offroad_nav2/docs/results/nolidar_seed*.json
```

Each run takes about 6 minutes. Results (score, touches, reversing, goals failed, bag check) are written to `ros2_ws/src/offroad_nav2/docs/results/`.

## 5. Look at what it does

- Topics worth echoing: `/perception/points/{geometric,nondrivable,clear,vehicle,vru,manmade,natural}`, `/perception/objects`,
  `/perception/latency_ms`, `/odometry/global`, `/local_costmap/costmap`, `/global_costmap/costmap`, `/cmd_vel`.
- Segmentation overlay: `/perception/seg_vis` (green drivable, magenta natural, yellow man-made, red VRU, blue vehicle).
- To see them in rviz run rviz inside the container (the X socket is mounted):
  `docker exec -it nav_sim bash -lc 'source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; rviz2'`
  and add the topics above; fixed frame `map`.
- Costmap and speed checks: `tools/costmap_check.py`, `tools/tl_test.py`, `tools/save_frames.py` under `ros2_ws/src/offroad_nav2/tools/`.

## 6. Stop

```bash
docker rm -f nav_sim
```

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Webots world looks white / no textures | remote assets were still loading. The `nav2ai` branch has them vendored; if it still happens, restart once more |
| `restart.sh` says `nav did not become active ... retrying once` | known Nav2 lifecycle bring-up race (lost DDS response). It retries automatically; if both tries fail run it again |
| Car does not move after a goal, `no valid path` in the log | a forward-only car can be trapped near a wall or barrel (no reverse by design). Teleport it back with `teleport.py`, or restart. Nav2 tuning is not finished |
| `cannot open display` or black Webots | run `xhost +local:` on the host; check `echo $DISPLAY` |
| `CUDA` errors in `perception_node` | GPU not passed in: the script uses `--gpus all`; check `nvidia-smi` on the host and the toolkit |
| Checkpoint not found | files must be in `ml_models_snapshot/` (mounted read-only as `/models`) |
| `/lidar` missing | intended: the lidar was removed from the world (commit "Remove the VLP-16" on the `ai`/`nav2ai` branch) |
| Another ROS 2 system on the host or network interferes | use another domain: `ROS_DOMAIN_ID=7 NAME=nav_sim ros2_ws/src/offroad_nav2/docker/run.sh '...'` |

## 8. Where things are

| What | Path |
|---|---|
| Launch files `launch/`, Nav2 parameters `offroad_bringup/nav2_params.py`, stereo/EKF configs `config/`, behaviour trees `behavior_trees/` | `ros2_ws/src/offroad_nav2/` |
| Perception, engine, geometry | `ros2_ws/src/offroad_nav2/` |
| Vehicle numbers | `ros2_ws/src/offroad_nav2/config/vehicle.yaml` |
| Zero-clamp relay, DBW stub | `ros2_ws/src/offroad_nav2/` |
| Simulator changes (stereo pair, no lidar, driver) | same package and branch (`ros2_ws/src/webots_ros2_suv/`) |
| Results, report, architecture | `ros2_ws/src/offroad_nav2/docs/REPORT.md`, `README.md`, `INTEGRATION_GUIDE.md`, `results/` |
