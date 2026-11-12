#!/usr/bin/env bash
set -e
set +u
source /opt/ros/humble/setup.bash
source /opt/carla_ros_ws/install/setup.bash

if [ -d /ros2_ws/src ] && [ ! -f /ros2_ws/install/setup.bash ]; then
    echo ""
    echo "RU: Первая сборка воркспейса, это займёт пару минут..."
    echo "EN: First workspace build, this takes a couple of minutes..."
    echo ""
    cd /ros2_ws && colcon build --symlink-install
fi

[ -f /ros2_ws/install/setup.bash ] && source /ros2_ws/install/setup.bash
set -u

if [ -t 1 ]; then
    echo ""
    echo "  ROS 2 ${ROS_DISTRO}  |  CARLA ${CARLA_HOST}:${CARLA_PORT}  |  ${RMW_IMPLEMENTATION}"
    echo "  ros2 launch carla_suv carla.launch.py"
    echo ""
fi

exec "$@"
