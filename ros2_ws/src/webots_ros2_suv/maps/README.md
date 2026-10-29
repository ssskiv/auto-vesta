# Map

`robocross.pgm` + `robocross.yaml`: occupancy grid of the track, 0.2 m per cell, origin (-31.4, -113) m.

Frame: the map was recorded starting from the spawn point of `worlds/robocross.wbt`, so the map origin is the
car's first GPS fix (`odom` origin) up to the spawn offset: world = map + (2.2, -1.6) m (see `WORLD_FROM_MAP` in
`vesta_autoware/scenario.py` and `webots_ros2_suv/referee.py`).

Used by:
- `ros2 run vesta_autoware make_maps` - turns it into the Autoware point cloud map and lanelet map;
- `ros2 run vesta_autoware make_lap` - plans the lap route away from the fence;
- the bridge (`map_yaml`) - lidar returns on occupied cells are the fence, not objects;
- the referee - splits the track into the Main and the Additional zone.

Rebuild the package (`colcon build`) after changing the map so it lands in `share/`.
