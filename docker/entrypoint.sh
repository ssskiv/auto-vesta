#!/usr/bin/env bash
set -e
set +u
source /opt/ros/humble/setup.bash

# RU: Собираем, если воркспейса нет или в нём не хватает пакета из src (например, после смены ветки).
# EN: Build when there is no workspace or a package from src is missing in it (e.g. after switching branches).
rm -f /tmp/.entrypoint_ready   # `make up` waits for this marker (the entrypoint build must not overlap with its own colcon build)
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
    # RU: На хостах с ОЗУ меньше 24 ГиБ собираем пакеты последовательно.
    # EN: Build packages sequentially on hosts with less than 24 GiB of RAM.
    build_args=(--symlink-install)
    host_mem_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
    if [ "$host_mem_kib" -lt $((24 * 1024 * 1024)) ]; then
        build_args+=(--executor sequential)
        echo ""
        echo "RU: На вашем ПК меньше 24ГБ. Собираем пакеты последовательно"
        echo "EN: Your PC have less than 24GB of RAM. Building packages sequentially"
        echo ""
    fi
    cd /ros2_ws && colcon build "${build_args[@]}"
fi

[ -f /ros2_ws/install/setup.bash ] && source /ros2_ws/install/setup.bash
set -u

if [ -t 1 ]; then
    echo ""
    echo "  ROS 2 ${ROS_DISTRO}  |  Webots R2025a  |  ${RMW_IMPLEMENTATION}"
    echo ""
fi

touch /tmp/.entrypoint_ready
exec "$@"
