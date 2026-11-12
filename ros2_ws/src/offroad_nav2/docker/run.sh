#!/usr/bin/env bash
# Run a command in vesta-nav with GPU + X, both repos mounted. Usage: run.sh <cmd...>
# /vesta = auto-vesta worktree (webots_ros2_suv + offroad_nav2), /offroad = autoware.off-road (only its Models/ directory is used), /models = checkpoint snapshots (ro),
# /ws = persistent colcon build/install/log (host ~/.cache/vesta-nav). Source /ws/install/setup.bash after build.sh.
xhost +local: >/dev/null 2>&1
docker run --rm --gpus all --ipc=host --net=host --shm-size=2g \
  -v /home/skiv/Programming/autoware.off-road:/offroad \
  -v /home/skiv/Programming/auto-vesta-ai:/vesta \
  -v /home/skiv/autoware_data/ml_models_snapshot:/models:ro \
  -v "$HOME/.cache/vesta-nav":/ws \
  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY="${DISPLAY:-:0}" -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e PYTHONUNBUFFERED=1 ${DOCKER_EXTRA:-} \
  ${NAME:+--name $NAME} vesta-nav:latest bash -lc "source /opt/ros/humble/setup.bash; [ -f /ws/install/setup.bash ] && source /ws/install/setup.bash; $*"
