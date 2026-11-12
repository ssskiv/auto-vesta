#!/usr/bin/env bash
set -e
set +u
source /opt/ros/humble/setup.bash

# RU: Собираем, если воркспейса нет или в нём не хватает пакета из src (например, после смены ветки).
# EN: Build when there is no workspace or a package from src is missing in it (e.g. after switching branches).
need_build=0
[ -f /ros2_ws/install/setup.bash ] || need_build=1
for pkg in webots_ros2_suv offroad_nav2; do
    if [ -d /ros2_ws/src/$pkg ] && [ ! -f /ros2_ws/install/$pkg/share/$pkg/package.xml ]; then need_build=1; fi
done
if [ -d /ros2_ws/src ] && [ "$need_build" = 1 ]; then
    echo ""
    echo "RU: Сборка воркспейса, это займёт пару минут..."
    echo "EN: Building the workspace, this takes a couple of minutes..."
    echo ""
    cd /ros2_ws && colcon build --symlink-install
fi

[ -f /ros2_ws/install/setup.bash ] && source /ros2_ws/install/setup.bash
set -u

if [ -t 1 ]; then
    echo ""
    echo "  ROS 2 ${ROS_DISTRO}  |  Webots R2025a  |  ${RMW_IMPLEMENTATION}"
    echo ""
fi

exec "$@"
