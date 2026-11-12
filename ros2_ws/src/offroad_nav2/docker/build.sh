#!/usr/bin/env bash
# colcon build of the Webots package and of offroad_nav2 (both from the auto-vesta ai/nav2ai worktree).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$HERE/run.sh" 'cd /ws && colcon build --symlink-install --build-base /ws/build --install-base /ws/install --base-paths /vesta/ros2_ws/src/webots_ros2_suv /vesta/ros2_ws/src/offroad_nav2 --event-handlers console_direct- 2>&1 | tail -25'
