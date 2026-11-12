#!/usr/bin/env python3
"""Host-side load sampler for the Nav2 + perception demo (processes of the Docker container are visible on the host).

Samples once per second: CPU % per role (from /proc/<pid>/stat, 100 % = one core), total CPU, GPU utilisation and memory
(nvidia-smi), and the per-process GPU SM utilisation (nvidia-smi pmon). Prints a table of mean / p95 per role for one
window. Usage: measure_load.py --label NAME --duration 40 [--json out.json]
Roles are matched on the command line, so the same script works for the Python and the C++ implementation."""
import argparse
import json
import os
import re
import subprocess
import time

ROLES = [  # (role, regex on the full command line)
    ('perception_node', r'(^|/)perception_node( |$)'),
    ('stamp_relay', r'(^|/)stamp_relay( |$)'),
    ('disparity_to_depth', r'(^|/)disparity_to_depth( |$)'),
    ('stereo SGBM container', r'component_container'),
    ('other offroad_nav2 nodes', r'(^|/)(sim_sensor_adapter|wheel_odometry|zero_clamp_relay|stop_wall_node|traffic_light_node)( |$)'),
    ('Nav2 (all servers)', r'(controller_server|planner_server|bt_navigator|behavior_server|waypoint_follower|velocity_smoother|collision_monitor|map_server|lifecycle_manager)'),
    ('localization (ekf/navsat)', r'(ekf_node|navsat_transform)'),
    ('Webots + drivers', r'(webots|ros2_supervisor|referee|vesta_driver|webots_controller)'),
]
HZ = os.sysconf('SC_CLK_TCK')


def procs():
    out = {}
    for pid in filter(str.isdigit, os.listdir('/proc')):
        try:
            cmd = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace').strip()
            st = open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()
            out[int(pid)] = (cmd, int(st[11]) + int(st[12]))   # utime + stime
        except Exception:
            continue
    return out


def role_of(cmd):
    if 'measure_load' in cmd or 'nvidia-smi' in cmd or not cmd:
        return None
    for name, rx in ROLES:
        if re.search(rx, cmd):
            return name
    return None


def total_cpu():
    f = open('/proc/stat').readline().split()[1:]
    v = list(map(int, f))
    return sum(v) - v[3] - v[4], sum(v)


def gpu():
    q = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True).stdout.strip().split(',')
    return float(q[0]), float(q[1]), float(q[2])


def gpu_procs():
    r = subprocess.run(['nvidia-smi', 'pmon', '-c', '1', '-s', 'um'], capture_output=True, text=True).stdout
    res = {}
    for line in r.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        p = line.split()
        try:
            res[int(p[1])] = (float(p[3]) if p[3] != '-' else 0.0, float(p[7]) if p[7] != '-' else 0.0)  # sm %, fb MiB
        except Exception:
            pass
    return res


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='run')
    ap.add_argument('--duration', type=float, default=40)
    ap.add_argument('--json')
    a = ap.parse_args()
    prev, prev_t = procs(), time.time()
    cpu0 = total_cpu()
    series = {r: [] for r, _ in ROLES}
    series.update({'TOTAL host CPU (% of all cores)': [], 'GPU util %': [], 'GPU mem MiB': [], 'GPU power W': [],
                   'perception_node GPU SM %': [], 'perception_node GPU mem MiB': [], 'Webots GPU SM %': []})
    ncpu = os.cpu_count()
    t_end = time.time() + a.duration
    while time.time() < t_end:
        time.sleep(1.0)
        cur, now = procs(), time.time()
        dt = now - prev_t
        per = {r: 0.0 for r, _ in ROLES}
        pid_role = {}
        for pid, (cmd, ticks) in cur.items():
            role = role_of(cmd)
            if role and pid in prev:
                per[role] += (ticks - prev[pid][1]) / HZ / dt * 100
            if role:
                pid_role[pid] = role
        for r in per:
            series[r].append(per[r])
        b, t = total_cpu()
        series['TOTAL host CPU (% of all cores)'].append(100.0 * (b - cpu0[0]) / max(1, t - cpu0[1]))
        cpu0 = (b, t)
        u, m, w = gpu()
        series['GPU util %'].append(u); series['GPU mem MiB'].append(m); series['GPU power W'].append(w)
        gp = gpu_procs()
        series['perception_node GPU SM %'].append(sum(v[0] for p, v in gp.items() if pid_role.get(p) == 'perception_node'))
        series['perception_node GPU mem MiB'].append(sum(v[1] for p, v in gp.items() if pid_role.get(p) == 'perception_node'))
        series['Webots GPU SM %'].append(sum(v[0] for p, v in gp.items() if pid_role.get(p) == 'Webots + drivers'))
        prev, prev_t = cur, now
    print(f'== {a.label}: {a.duration:.0f} s window, {ncpu} logical CPUs (CPU % of one core, so 100 = one full core)')
    print(f'{"metric":38s} {"mean":>8s} {"p95":>8s} {"max":>8s}')
    out = {}
    for k, v in series.items():
        if not v:
            continue
        mean = sum(v) / len(v)
        print(f'{k:38s} {mean:8.1f} {pct(v, 0.95):8.1f} {max(v):8.1f}')
        out[k] = {'mean': round(mean, 2), 'p95': round(pct(v, 0.95), 2), 'max': round(max(v), 2)}
    if a.json:
        json.dump({'label': a.label, 'duration': a.duration, 'metrics': out}, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
