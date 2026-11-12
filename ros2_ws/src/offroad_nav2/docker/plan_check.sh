#!/usr/bin/env bash
# Offline planner check of the lap route (no Webots): static map + Nav2 only.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$HERE/build.sh" 2>&1 | grep -E "Failed|rror"
"$HERE/run.sh" 'ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --yaw 0 --frame-id map --child-frame-id odom > /dev/null 2>&1 &
ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --yaw 0 --frame-id odom --child-frame-id base_link > /dev/null 2>&1 &
ros2 launch offroad_nav2 navigation.launch.py obstacles:=${OBST:-camera} > /tmp/nav.log 2>&1 &
sleep 40; python3 /vesta/ros2_ws/src/offroad_nav2/tools/plan_check.py ${ROUTE:-/vesta/ros2_ws/src/offroad_nav2/config/lap_route.json}'
