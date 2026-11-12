#!/usr/bin/env python3
"""Before/after table from docs/perf/{python,cpp}_{idle,drive}_{load,topics}.json -> docs/perf/PERF_REPORT.md"""
import json
import os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'perf')


def load_topics(name):
    """parse the text output of measure_topics.py (the JSON stayed inside the container)"""
    import ast
    import re
    p = os.path.join(D, name.replace('.json', '.txt'))
    if not os.path.exists(p):
        return None
    txt = open(p).read()
    rates = {m.group(1): float(m.group(2)) for m in re.finditer(r'^(/\S+)\s+([\d.]+) Hz', txt, re.M)}
    m = re.search(r'latency_ms (\{.*\})', txt)
    return {'rates_hz': rates, 'latency_ms': ast.literal_eval(m.group(1))} if m else None


def load(name):
    p = os.path.join(D, name)
    return json.load(open(p)) if os.path.exists(p) else None


rows = ['perception_node', 'stamp_relay', 'disparity_to_depth', 'stereo SGBM container', 'other offroad_nav2 nodes', 'Nav2 (all servers)',
        'localization (ekf/navsat)', 'Webots + drivers', 'TOTAL host CPU (% of all cores)', 'GPU util %', 'GPU mem MiB', 'GPU power W',
        'perception_node GPU SM %', 'Webots GPU SM %']
out = ['# Performance: Python vs C++ (measured with tools/measure_load.py + tools/measure_topics.py)\n',
       'CPU % are of ONE core (100 = one full core); 32 logical CPUs. Same machine, same demo (Webots, stereo, Nav2), same barrel seed.\n']
for phase, title in (('idle', 'Car stationary (40 s window)'), ('drive', 'Car driving a 95 m goal at 3 m/s (45 s window)')):
    py, cp = load(f'python_{phase}_load.json'), load(f'cpp_{phase}_load.json')
    pt, ct = load_topics(f'python_{phase}_topics.json'), load_topics(f'cpp_{phase}_topics.json')
    if not (py and cp):
        continue
    out += [f'\n## {title}\n', '| metric (mean) | Python | C++ | change |', '|---|---:|---:|---:|']
    for r in rows:
        a, b = py['metrics'].get(r, {}).get('mean'), cp['metrics'].get(r, {}).get('mean')
        if a is None or b is None:
            continue
        ch = f'{100 * (b - a) / a:+.0f} %' if a else ''
        out.append(f'| {r} | {a:.1f} | {b:.1f} | {ch} |')
    if pt and ct:
        out += ['', '| end-to-end | Python | C++ |', '|---|---:|---:|']
        for k in ('p50', 'p95', 'max'):
            out.append(f'| image-to-cloud latency {k} (ms) | {pt["latency_ms"][k]} | {ct["latency_ms"][k]} |')
        out.append(f'| cloud rate (Hz) | {pt["rates_hz"].get("/perception/points/geometric")} | {ct["rates_hz"].get("/perception/points/geometric")} |')
        out.append(f'| depth rate (Hz) | {pt["rates_hz"].get("/stereo/depth")} | {ct["rates_hz"].get("/stereo/depth")} |')
open(os.path.join(D, 'PERF_REPORT.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
