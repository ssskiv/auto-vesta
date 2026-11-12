#!/usr/bin/env bash
# Referee runs: for each seed restart the whole demo with that barrel layout, drive one lap (lap_run), record a /cmd_vel
# bag, store the result in ros2_ws/src/offroad_nav2/docs/results/. Usage: run_seeds.sh "1 2 3 4 5" [launch args, e.g. tl_source:=standin]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS=${1:-"1 2 3 4 5"}; shift
TAG=${TAG:-camera}
OUT=/vesta/ros2_ws/src/offroad_nav2/docs/results
for s in $SEEDS; do
  echo "=== seed $s ($TAG) $(date +%T)"
  LOG=/tmp/nav_sim_${TAG}_$s.log "$HERE/restart.sh" barrels_seed:=$s "$@" || continue
  sleep 25
  docker exec nav_sim bash -lc "source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; rm -rf /tmp/bag_$s; (ros2 bag record -o /tmp/bag_$s /cmd_vel > /dev/null 2>&1 &); sleep 2; ros2 run offroad_nav2 lap_mission --timeout 900 --out $OUT/${TAG}_seed$s.json > /tmp/lap_$s.log 2>&1; pkill -INT -f 'ros2 bag record' ; sleep 3; python3 /vesta/ros2_ws/src/offroad_nav2/tools/bag_cmd_vel_check.py /tmp/bag_$s | tee $OUT/${TAG}_seed${s}_bag.txt; python3 - <<PY
import json; r=json.load(open('$OUT/${TAG}_seed$s.json')); f=r['referee']; print('goals_failed',r.get('goals_failed'),'penalty',f['penalty'],'total',f['total'],'finished',f['finished'], 'events', [e['text'] for e in f['events'] if e['points']<0])
PY"
done
