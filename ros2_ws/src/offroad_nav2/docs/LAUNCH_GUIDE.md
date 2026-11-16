# How to launch the Vesta perception + Nav2 demo (Webots, no lidar)

Everything runs in Docker (ROS 2 Humble). Nothing is installed on the host. The simulated car has **no lidar**: obstacles come only
from the stereo camera and the three networks.

## 1. Prerequisites (once)

| Needed | Check |
|---|---|
| Linux `amd64`, NVIDIA GPU, compatible host driver, and NVIDIA Container Toolkit for full perception | `docker run --rm --gpus all ubuntu nvidia-smi` |
| An X display on the host (Webots camera rendering needs OpenGL, even with `gui:=false`) | `echo $DISPLAY` (`:0`) |
| Public image `ghcr.io/ssskiv/vesta-sim:nav2ai` | `docker image inspect ghcr.io/ssskiv/vesta-sim:nav2ai` |
| An auto-vesta checkout on branch `nav2ai` (this package lives in `ros2_ws/src/offroad_nav2/`) | From its root: `git branch --show-current` |
| Local network definitions included in auto-vesta | `ls ros2_ws/src/offroad_nav2/tools/model_components/` |
| Checkpoint copies in the checkout's `ml_models/` directory (`last_freespaceseg.pth`, `last_freespacecontour.pth`, `last_objectseg.pth`) | `ls ml_models/` |

Run commands from the root of your auto-vesta checkout. Its location is unrestricted.
Create the local checkpoint directory:

```bash
mkdir -p ml_models
```

Copy completed training checkpoints into `ml_models/`; keep this directory separate from live training output.
The folder is ignored by Git. For example, with `TRAINING_MODELS` set to your training output directory:

```bash
cp "$TRAINING_MODELS/ObjectSeg/last_objectseg.pth" ml_models/
```

`make onnx` uses the vendored `tools/model_components/` definitions to construct the networks before loading
the checkpoint weights. No external model repository or repository-path variables are needed.
The C++ perception runtime reads the exported `ml_models/onnx/*.onnx` files.

The published image is `ghcr.io/ssskiv/vesta-sim:nav2ai`. Public pulls do not
require GHCR login. To acquire it without starting containers, run `make image`:
this uses the cached image when available and downloads it only when missing.
Use `make pull` to explicitly refresh it from the registry. For a local,
incremental image build use `make image-build`; `make rebuild` rebuilds the
local `vesta-sim:latest` without cache and clears `.build`.

For a direct Compose command, source the shared image selector in that same
operator shell. `make` cannot export a variable back to its caller:

```bash
source docker/image.sh
# Optional: select the locally built image
export VESTA_IMAGE=vesta-sim:latest
docker compose -f docker/docker-compose.yml -f docker/compose.nvidia.yml config
```

To pin or roll back to the verified bootstrap image, use its observed digest:

```bash
export VESTA_IMAGE=ghcr.io/ssskiv/vesta-sim@sha256:cf0a98f3e5544443acee5c4e0e95dfdd6349c9d653b931589762435bc56c6524
make image
```

Image tags can move; this digest identifies the verified bootstrap image.
The image targets `linux/amd64`. Full perception needs an NVIDIA GPU, a
compatible host driver, and NVIDIA Container Toolkit. A CPU dependency smoke
check does not prove GPU execution or Webots rendering. Model checkpoints and
the external `autoware.off-road/Models` source are mounted separately from the
image, and ROS workspace builds run at container startup and incrementally
through `make up`. Changing or pinning the image does not roll back mounts or
build caches.

Publishing is a maintainer operation: it requires GHCR login with a secure
package-publishing PAT, package visibility set to Public, the package connected
to this repository, and appropriate repository Actions access. Anonymous
consumer pulls do not require those publisher credentials.

## 2. Build

All runtime nodes of `offroad_nav2` are **C++** (`src/nodes/`, shared pure-C++ core in `src/core/`: geometry, post-processing, tracking, vehicle logic,
ONNX Runtime engine). Only the launch files, the Nav2 parameter generator and the operator tools (`nav_goal`, `gps_route`, `lap_*`, `tools/`) are Python.
The three networks run through ONNX Runtime (CUDA); the PyTorch checkpoints are converted once:

```bash
make onnx       # checkpoints (*.pth in the snapshot dir) -> <snapshot>/onnx/{seg,contour,objseg}.onnx (fp16, normalisation + argmax inside)
```

Run `make onnx` once and again whenever the checkpoints change (`make up` warns when the files are missing).

The shared Docker image contains Webots, ROS 2 Humble, Nav2, CUDA-enabled
perception dependencies and stereo support. The default is the public
`ghcr.io/ssskiv/vesta-sim:nav2ai`; `make image` acquires and validates it
without starting a container. Local builds use the separate tag
`vesta-sim:latest`.

```bash
make image      # acquire and validate the selected image
```

### Interactive: `make up` (recommended)

```bash
make up         # acquire image if missing; start vesta_sim, build workspace, open bash
ros2 launch offroad_nav2 webots_nav.launch.py        # inside the container
```

The container keeps running when you leave bash. Every `make up` re-runs the (incremental) `colcon build` and sources the workspace, and
`make exec` / `make attach` opens another shell without building. The checkpoint snapshot directory (`ml_models/` in this checkout, overridden with `VESTA_MODELS=`)
is mounted read-only as `/models`. Model definitions are included in the mounted auto-vesta source tree.

The scripted helpers use the same `ml_models/` default and honor `VESTA_MODELS`.

Webots controller and plugin links are generated locally when the simulator launches and refreshed after an
installation move. They use `WEBOTS_HOME` (the Docker image sets it to `/usr/local/webots`).
For an installation outside Docker, set `WEBOTS_HOME` to its directory before launching or running
`ros2_ws/src/webots_ros2_suv/tools/vendor_webots_assets.py`.

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
| Checkpoint not found | put checkpoint copies in this checkout's `ml_models/` directory, or set `VESTA_MODELS` (mounted read-only as `/models`) |
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
