# Map

`robocross.pgm` + `robocross.yaml`: occupancy grid of the track, 0.2 m per cell, origin (-31.4, -113) m.

Frame: the map was recorded starting from the spawn point of `worlds/robocross.wbt`, so the map origin is the
car's first GPS fix (`odom` origin) up to the spawn offset: world = map + (2.2, -1.6) m (see `WORLD_FROM_MAP` in
`webots_ros2_suv/referee.py`).

Used by:
- the referee - splits the track into the Main and the Additional zone;
- `offroad_nav2` - the static layer of the global costmap (the fence is occupied cells) and the lap route
  (`ros2_ws/src/offroad_nav2/tools/make_lap_route.py`).

Rebuild the package (`colcon build`) after changing the map so it lands in `share/`.
