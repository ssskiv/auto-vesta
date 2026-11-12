# How to launch the Vesta perception + Nav2 demo (Webots, no lidar)

Everything runs in Docker (ROS 2 Humble). Nothing is installed on the host. The simulated car has **no lidar**: obstacles come only
from the stereo camera and the three networks.

## 1. Prerequisites (once)

| Needed | Check |
|---|---|
| Docker with the NVIDIA container toolkit | `docker run --rm --gpus all ubuntu nvidia-smi` |
| An X display on the host (Webots camera rendering needs OpenGL, even with `gui:=false`) | `echo $DISPLAY` (`:0`) |
| Base image `vesta-sim:latest` (the auto-vesta image) | `docker images vesta-sim` |
| auto-vesta repo at `/home/skiv/Programming/auto-vesta` on branch `nav2ai` (this package lives in it: `ros2_ws/src/offroad_nav2/`) | `git -C /home/skiv/Programming/auto-vesta branch --show-current` |
| The `autoware.off-road` repo (only its `Models/` directory is used, mounted read-only as `/offroad`) | `ls /home/skiv/Programming/autoware.off-road/Models` |
| Checkpoint copies in `/home/skiv/autoware_data/ml_models_snapshot/` (`last_freespaceseg.pth`, `last_freespacecontour.pth`, `last_objectseg.pth`) | `ls /home/skiv/autoware_data/ml_models_snapshot` |

If you are on another branch: `cd /home/skiv/Programming/auto-vesta && git checkout nav2ai` (your earlier uncommitted sim work is saved on the branch `sim-wip`).
To refresh the ObjectSeg snapshot while training runs, copy the file; never point the demo at the live `ml_models/` folder:
`cp /home/skiv/autoware_data/ml_models/ObjectSeg/last_objectseg.pth /home/skiv/autoware_data/ml_models_snapshot/`.

All commands below are run from `/home/skiv/Programming/auto-vesta` (the auto-vesta checkout on branch `nav2ai`). The paths inside the scripts are absolute, so the
repositories must stay where they are.

## 2. Build

All runtime nodes of `offroad_nav2` are **C++** (`src/nodes/`, shared pure-C++ core in `src/core/`: geometry, post-processing, tracking, vehicle logic,
ONNX Runtime engine). Only the launch files, the Nav2 parameter generator and the operator tools (`nav_goal`, `gps_route`, `lap_*`, `tools/`) are Python.
The three networks run through ONNX Runtime (CUDA); the PyTorch checkpoints are converted once:

```bash
make onnx       # checkpoints (*.pth in the snapshot dir) -> <snapshot>/onnx/{seg,contour,objseg}.onnx (fp16, normalisation + argmax inside)
```

Run `make onnx` once and again whenever the checkpoints change (`make up` warns when the files are missing).

There is one Docker image (`vesta-sim:latest`: Webots, ROS 2 Humble, Nav2, PyTorch/CUDA, stereo). `make` builds it only when it is
missing or older than the Nav2 + perception layers (it checks the label `vesta.nav`); the first build takes 15-25 minutes.

```bash
make image      # build the image if needed (without starting anything)
```

### Interactive: `make up` (recommended)

```bash
make up         # build the image if needed, start the container vesta_sim in the background,
                # run colcon build, source install/setup.bash and open bash
ros2 launch offroad_nav2 webots_nav.launch.py        # inside the container
```

The container keeps running when you leave bash. Every `make up` re-runs the (incremental) `colcon build` and sources the workspace, and
`make exec` / `make attach` opens another shell without building. The checkpoints (`~/autoware_data/ml_models_snapshot`, `VESTA_MODELS=`)
and `Models/` of autoware.off-road (`~/Programming/autoware.off-road`, `OFFROAD_REPO=`) are mounted read-only as `/models` and `/offroad`.

### Scripted: the helpers under `ros2_ws/src/offroad_nav2/docker/`

`make demo` (or `docker/restart.sh`) builds into a separate cache (`~/.cache/vesta-nav`), starts the whole demo in the container
`nav_sim` and waits until Nav2 is active; `build.sh` only builds.

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
| `barrels_seed` | `7` | `>0` fixed barrel layout, `0` leave the world random |
| `tl_source` | `none` | `standin` = traffic light state from the world (a labelled stand-in, not perception) and a stop wall at the stop line |

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
TAG=run ros2_ws/src/offroad_nav2/docker/run_seeds.sh "1 2 3 4 5"      # restarts the demo per seed, records a /cmd_vel bag
ls ros2_ws/src/offroad_nav2/docs/results/run_seed*.json
```

Each run takes about 6 minutes. Results (score, touches, reversing, goals failed, bag check) are written to `ros2_ws/src/offroad_nav2/docs/results/`.

## 5. Look at what it does

- Topics worth echoing: `/perception/points/{geometric,nondrivable,clear,vehicle,vru,manmade,natural}`, `/perception/objects`,
  `/perception/latency_ms`, `/odometry/global`, `/local_costmap/costmap`, `/global_costmap/costmap`, `/cmd_vel`.
- Segmentation overlay: `/perception/seg_vis` (green drivable, magenta natural, yellow man-made, red VRU, blue vehicle).
- rviz: `docker exec -it nav_sim bash -lc 'source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; rviz2'`. It opens with the
  default config `ros2_ws/src/rviz/default.rviz` (mounted as `~/.rviz2`, also by `make shell`): fixed frame `map`, robot model, TF,
  global/local costmaps, plan, odometry, the perception clouds (red = depth obstacles, blue vehicle, magenta people, yellow man-made,
  green natural, orange non-drivable), stop/slow zones, the left camera image and the segmentation overlay. Use "2D Goal Pose" to send a goal.
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
| Another ROS 2 system on the host or network interferes | use another domain: `ROS_DOMAIN_ID=7 NAME=nav_sim ros2_ws/src/offroad_nav2/docker/run.sh '...'` |

## 8. Where things are

| What | Path |
|---|---|
| Launch files `launch/`, all parameters in YAML under `config/` (`nodes.yaml` for the C++ nodes, `nav2_params.yaml`, `ekf.yaml`, `navsat.yaml`, `stereo.yaml`, `vehicle.yaml`), behaviour trees `behavior_trees/` | `ros2_ws/src/offroad_nav2/` |
| Perception, engine, geometry | `ros2_ws/src/offroad_nav2/` |
| Vehicle numbers | `ros2_ws/src/offroad_nav2/config/vehicle.yaml` |
| Zero-clamp relay, DBW stub | `ros2_ws/src/offroad_nav2/` |
| Simulator changes (stereo pair, no lidar, driver) | same package and branch (`ros2_ws/src/webots_ros2_suv/`) |
| Results, report, architecture | `ros2_ws/src/offroad_nav2/docs/REPORT.md`, `README.md`, `INTEGRATION_GUIDE.md`, `results/` |
