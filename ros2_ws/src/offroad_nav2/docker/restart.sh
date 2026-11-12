#!/usr/bin/env bash
# Rebuild, restart the whole demo in container nav_sim, wait until Nav2 is active. Usage: restart.sh [launch args...]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker rm -f nav_sim >/dev/null 2>&1
"$HERE/build.sh" 2>&1 | grep -E "Failed|rror"
LOG=${LOG:-/tmp/nav_sim.log}
(NAME=nav_sim "$HERE/run.sh" "ros2 launch offroad_nav2 webots_nav.launch.py $*" > "$LOG" 2>&1 &)
for i in $(seq 1 30); do
  sleep 5
  timeout 25 docker exec nav_sim bash -lc 'source /opt/ros/humble/setup.bash; for n in controller_server bt_navigator planner_server; do timeout 5 ros2 lifecycle get /$n 2>/dev/null | grep -q active || exit 1; done' 2>/dev/null && { echo "nav active after $((i*5)) s"; exit 0; }
done
echo "nav did not become active (lifecycle bring-up race); retrying once"
if [ -z "$RETRIED" ]; then RETRIED=1 exec "$0" "$@"; fi
exit 1
