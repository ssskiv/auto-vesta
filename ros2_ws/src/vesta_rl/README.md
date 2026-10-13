# vesta_rl: RL local planner + controller (Yandex-rover scheme)

One network drives the car: it gets the local costmap, the global path and
`/odom`, and outputs `cmd_vel` (speed + steering). It runs inside the Nav2
plugin `vesta_neural_controller` next to a classical controller (RPP). Each
tick both commands are rolled out 2 s on the local costmap and the better
safe one is sent; if neither is safe the car stops.

```
planner_server (SmacHybrid) ── global path ──┐
/odom, local_costmap ──► NeuralController (controller_server)
                          ├─ policy.onnx ─► (v, δ)_nn
                          ├─ RPP ─────────► (v, δ)_rpp
                          └─ rollouts → NN / RPP / STOP ─► /cmd_vel ─► vesta_driver
```

| File | What |
|---|---|
| `config.py` | Shared constants: vehicle, observation, action mapping. Change together with `vesta_neural_controller/observation.hpp`. |
| `obs.py` | Observation spec (reference). The C++ port is tested against it. |
| `world.py`, `sim2d.py` | Fast 2D gym env: bicycle model with lag, lidar marking and Nav2 inflation, random obstacles unknown to the global path, domain randomization. |
| `procgen.py` | Procedural tracks: random drivable path, barriers (dashed / rows of objects, one or two sides, gaps), clutter. |
| `scene_pool.py` | Fixed pool of training scenes (maps x obstacle layouts), generated once and reviewed by a person before training. |
| `paths.py` | Path library (A* + smoothing + turning-radius check). `data/paths.npz` is pre-built for `maps/robocross.yaml`. |
| `reward.py` | Reward and episode end, shared by both envs. |
| `train.py` | PPO (stable-baselines3), optional BC warm start, Webots fine-tuning. |
| `export_onnx.py` | Exports the deterministic actor and checks ONNX against torch. |
| `webots_env.py` | Gym env over ROS + Webots (teleport resets via `/vesta/teleport`). |
| `calibrate.py` | Fits the 2D vehicle lag to Webots. |
| `viewer.py` | Window that shows the newest checkpoint driving in the 2D sim. |

## Workflow
All commands inside the container, from `/ros2_ws`, after `colcon build && source install/setup.bash`.

1. **Path library**, only after re-recording the map:
   ```bash
   ros2 run vesta_rl make_paths --map src/webots_ros2_suv/maps/robocross.yaml --out src/vesta_rl/data/paths.npz
   ```
2. **Calibrate the 2D sim against Webots.** Start `robot.launch.py`, put the car somewhere open, then run:
   ```bash
   ros2 run vesta_rl calibrate --out calib.csv
   ```
   Copy the fitted values into `VehicleParams` defaults (centre of `random()`) in `sim2d.py`, and into `tau_v/accel_max/tau_delta/steer_rate` of `FollowPath` in `nav2_params.yaml`.
3. **Generate and review the scene pool.** 100 maps of all kinds (25 routes on `robocross`, 75 procedural tracks: barriers on both sides dashed / objects / mixed, one side, open) x 200 obstacle layouts = 20 000 scenes:
   ```bash
   ros2 run vesta_rl scene_pool generate --out src/vesta_rl/data/pool \
       --map src/webots_ros2_suv/maps/robocross.yaml --paths src/vesta_rl/data/paths.npz
   ```
   Look at `data/pool/previews/mXXX_p0.png` (16 layouts per map, labelled `map/layout`; more pages with `scene_pool preview --pool ... --pages 1 2`) or step through the scenes:
   ```bash
   ros2 run vesta_rl scene_pool review --pool src/vesta_rl/data/pool   # x reject layout, m reject map, arrows, q
   ```
   Rejected scenes are listed in `data/pool/rejected.txt` (`17` = map 17, `17/5` = its layout 5; it can be edited by hand). Training only uses the rest.

   **Automatic check.** Every scene is accepted only if a Nav2-like planner (`feasibility.py`: Hybrid-A* with Dubins shots, forward only) finds a route with turning radius 5 m (car: 3.9 m), 1 m between the body and every obstacle and ghost, base_link within 4.5 m of the global path. Scenes without a route are thrown away and redrawn; a map is replaced if it rarely gets one. The route is stored with the scene: previews draw it in green, and the BC expert (`expert_action`) drives along it. Training does no checks.
4. **Train in the 2D sim.** Takes hours on a GPU. Watch `rollout/success_rate`:
   ```bash
   ros2 run vesta_rl train --pool src/vesta_rl/data/pool \
       --out runs/sim --steps 30000000 --bc-steps 100000 --device cuda --max-ram-gb 12
   tensorboard --logdir runs/sim/tb
   ```
   Stop at any time with Ctrl+C; `runs/sim/model.zip` is saved. Continue with `--resume runs/sim/model.zip`.

   **What the episodes look like:**
   - Each episode is a random non-rejected scene of the pool: boxes, barrels and wall pieces (up to 15, some right on the path) and sometimes ghost lidar marks. The global path doesn't know about them. Start pose, vehicle lag, lidar dropouts and sensor noise are still random per episode.
   - `--max-obstacles 15`: at most this many of the scene's obstacles (a random subset). The curriculum starts at 3 and raises it as the success rate grows.

   **Speed action.** The throttle action is a fraction of the path speed limit `obs.speed_limit` (same in C++ `speedLimit`): v = v_lim · (a_v + 1) / 2. The policy cannot exceed the curve or goal limit; it can only go slower, e.g. for obstacles. Models trained before this change interpret a_v as a fraction of 5 m/s.

   **Speed in curves and exploration:**
   - The reward penalizes driving faster than the curve allows. The limit keeps lateral acceleration within 2 m/s² for the path in the next 12 m, with room to brake at 2 m/s² (`reward.py: A_LAT_MAX, DECEL, K_OVERSPEED`). That's about 3 m/s in the tightest turn.
   - The same limit also makes the car slow down to 1 m/s at the end of the path (`V_GOAL`).
   - `--ent-coef 0.003` (default) is an entropy bonus: it keeps the policy trying different actions. At 0.01 the action std kept growing.
   - `--reset-std 0.55` with `--resume` raises the action randomness back up when it has dropped too early.

   **Curriculum.** Training starts easy and gets harder as the policy copes:
   - level 0 = up to 3 obstacles, 10% of them on the path;
   - level 1 = `--max-obstacles`, 35% on the path;
   - +0.1 when `success_rate` over the last 100 episodes reaches 0.6, at most once per 300k steps.

   The level is shown in tensorboard (`curriculum/level`) and saved to `<out>/curriculum.json`, so `--resume` continues from it. `--curriculum 1` turns it off; `--curriculum 0.5` starts from a given level.

   **CPU / RAM limits.** The defaults keep the PC usable:

   | Flag | Default | Effect |
   |---|---|---|
   | `--n-envs` | half the cores, max 8 | Sim worker processes. Each uses one core and about 200 MB. |
   | `--threads` | 4 | Torch threads in the main process. |
   | `--nice` | 10 | Lower CPU priority, so the desktop stays responsive. |
   | `--max-ram-gb` | off | Above this, the model is saved and training stops. |

   With 8 workers training uses about 2.5 GB in total; the BC buffer adds about 0.8 GB per 100k `--bc-steps`. Each worker also runs its BLAS/OpenMP with one thread.

   For a hard cap on the whole container (training + Webots), run this from the host:
   ```bash
   docker update --cpus 12 --memory 16g --memory-swap 16g webots_vesta
   ```
   **Watch the policy while it trains.** In a second terminal in the same container:
   ```bash
   ros2 run vesta_rl viewer --run runs/sim --pool src/vesta_rl/data/pool   # keys: space pause, n next episode, +/- speed, q quit
   ```
   It runs its own copy of the sim with the newest checkpoint, and reloads when a new one appears (every ~1M steps). It uses one core and doesn't affect training.
5. **Fine-tune in Webots** (optional, slow: real time):
   ```bash
   ros2 launch webots_ros2_suv robot.launch.py gui:=false     # terminal 1
   ros2 launch webots_ros2_suv nav2.launch.py                 # terminal 2
   ros2 run vesta_rl train --env webots --paths src/vesta_rl/data/paths.npz \
       --resume runs/sim/model.zip --out runs/webots --steps 300000
   ```
6. **Export and deploy:**
   ```bash
   ros2 run vesta_rl export_onnx --model runs/sim/model.zip --out src/vesta_neural_controller/models/policy.onnx
   colcon build --packages-select vesta_neural_controller
   ros2 launch webots_ros2_suv nav2.launch.py      # or model:=/abs/path/policy.onnx
   ```
7. **Evaluate in Webots.**
   - **Goals from RViz:** `nav2.launch.py` starts `goal_follower`. Click **2D Goal Pose** in RViz, and it plans a path (shown as "Global path") and drives it with the neural controller. A new click replaces the current goal. Turn it off with `goal_follower:=false`. Results are logged per goal: SUCCEEDED/ABORTED and the NN / RPP / STOP share.
   - **Goals from the terminal:** `ros2 run vesta_rl drive --library` or `--goal X Y`.
   - **Topics to watch:**
     - `/neural_controller/source` (NN / RPP / STOP): the share of NN is the "% under neural control" metric.
     - `/neural_controller/rollouts` (MarkerArray): the chosen rollout is green, a colliding one red.

## Changing the observation
Edit `config.py` / `obs.py` **and** `vesta_neural_controller/observation.hpp|cpp`, then regenerate the fixture and run the C++ test:
```bash
ros2 run vesta_rl make_obs_fixture --out src/vesta_neural_controller/test/fixtures/obs_cases.txt
colcon test --packages-select vesta_neural_controller vesta_rl && colcon test-result --verbose
```
Old models are incompatible after such a change.
