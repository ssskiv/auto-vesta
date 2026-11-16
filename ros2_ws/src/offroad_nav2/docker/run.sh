#!/usr/bin/env bash
# Run a command in vesta-sim with GPU + X and the checkout mounted. Usage: run.sh <cmd...>
# /vesta = auto-vesta repo root (webots_ros2_suv + offroad_nav2), /models = checkpoint snapshots (ro),
# /ws = persistent colcon build/install/log (host ~/.cache/vesta-nav). Source /ws/install/setup.bash after build.sh.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VESTA="$(cd "$HERE/../../../.." && pwd)"   # auto-vesta repo root (this file: ros2_ws/src/offroad_nav2/docker/)
source "$VESTA/docker/image.sh" || exit $?
vesta_image_acquire || exit $?
VESTA_MODELS="${VESTA_MODELS:-$VESTA/ml_models}"
# ros2_ws/src/rviz is mounted as ~/.rviz2, so `rviz2` starts with ros2_ws/src/rviz/default.rviz
xhost +local: >/dev/null 2>&1
docker run --pull=never --rm --gpus all --ipc=host --net=host --shm-size=2g \
  -v "$VESTA":/vesta \
  -v "$VESTA/ros2_ws/src/rviz":/root/.rviz2 \
  -v "$VESTA_MODELS":/models:ro \
  -v "$HOME/.cache/vesta-nav":/ws \
  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY="${DISPLAY:-:0}" -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e PYTHONUNBUFFERED=1 ${DOCKER_EXTRA:-} \
  ${NAME:+--name $NAME} "$VESTA_IMAGE" bash -lc "source /opt/ros/humble/setup.bash; [ -f /ws/install/setup.bash ] && source /ws/install/setup.bash; $*"
