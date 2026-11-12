#!/usr/bin/env bash
# =========================================================================
# RU: Запуск симуляции в CARLA. Те же флаги, что у start.sh.
# EN: Run the simulation in CARLA. Same flags as start.sh.
#
#     ./start_carla.sh              CARLA + контейнер ROS 2 с carla-ros-bridge
#     ./start_carla.sh --headless   CARLA без окна (-RenderOffScreen)
#     ./start_carla.sh --stop       остановить оба контейнера
#
#     RU: затем в оболочке контейнера:  ros2 launch carla_suv carla.launch.py
#     EN: then in the container shell:  ros2 launch carla_suv carla.launch.py
# =========================================================================
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM=carla exec "$HERE/start.sh" "$@"
