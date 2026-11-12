# CARLA map `Robocross`

The track exists as a Webots world only. The CARLA map has to be built once in Unreal and cooked into a package.

## Build the map

1. Build CARLA from source with the version pinned in `docker/compose.carla.yml` (`CARLA_VERSION`), UE4 fork
   `carla-simulator/UnrealEngine`.
2. Make the assets from the occupancy grid:
   ```bash
   python3 ros2_ws/src/carla_suv/tools/robocross_assets.py \
       --map-yaml ros2_ws/src/webots_ros2_suv/maps/robocross.yaml --out carla/assets
   ```
   `fence.obj` and `ground.obj` are in meters, CARLA axes. Import them into Unreal with a uniform scale of 100.
   `arena.xodr` is a flat drivable stand-in, not a road network: the track is an open field bounded by a fence.
3. Add the signs (`RobocrossSign` x14), the gates (`Winder*`) and the barrel mesh. Their positions are in
   `ros2_ws/src/webots_ros2_suv/worlds/robocross.wbt` (Webots axes: negate y for CARLA). Textures and meshes are in
   `ros2_ws/src/webots_ros2_suv/protos/`.
4. Spawn point: world (2.435, -2.261), yaw 0. Keep world = map + (2.2, -1.6) m.
5. Cook the map with `make package`, then put the resulting `*.tar.gz` into `carla/Import/`. The `carlasim`
   container imports it at start (`ImportAssets.sh`).

## Without the Unreal map

```bash
python3 ros2_ws/src/carla_suv/tools/robocross_assets.py --map-yaml ... --out carla/assets
ros2 launch carla_suv carla.launch.py xodr:=/ros2_ws/carla/assets/arena.xodr
```
(Mount `carla/` into the ROS container first.)
