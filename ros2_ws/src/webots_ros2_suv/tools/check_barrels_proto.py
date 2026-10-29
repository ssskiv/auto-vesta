#!/usr/bin/env python3
"""Render protos/RobocrossBarrels.proto outside Webots (node) and check the rules on the result.

The JavaScript is built the way Webots R2023b does it (src/webots/core/WbTemplateEngine.cpp:
text -> `___vrml += render(\`...\`)`, %<= e >% -> expression, %< s >% -> statement), with
wbrandom.js re-implemented below (same LCG as resources/web/wwi/.../modules/webots/wbrandom.js).

    python3 tools/check_barrels_proto.py [--seeds 20]
"""
import argparse
import json
import math
import os
import re
import subprocess
import tempfile

import cv2
import numpy as np

import place_barrels as P

WBRANDOM = """
let seedValue = 0;
export function seed(s) { seedValue = s; }
function updateSeed() { seedValue = (1140671485 * seedValue + 12820163) % Math.pow(2, 24); }
export function real(min, max) {
  if (typeof min === 'undefined') { min = 0; max = 1; } else if (typeof max === 'undefined') { max = min; min = 0; }
  updateSeed();
  return min + seedValue * (max - min) / (Math.pow(2, 24) - 1);
}
export function integer(min, max) {
  if (typeof min === 'undefined') { min = 0; max = Math.pow(2, 24) - 1; } else if (typeof max === 'undefined') { max = min; min = 1; }
  updateSeed();
  return min + seedValue % (1 + max - min);
}
"""


def to_js(proto):
    body = proto[proto.index(']\n{') + 3:proto.rstrip().rindex('}')]
    js, pos = [], 0
    while True:
        a = body.find('%<', pos)
        if a < 0:
            js.append('___vrml += render(`' + body[pos:] + '`);')
            break
        js.append('___vrml += render(`' + body[pos:a] + '`);')
        b = body.index('>%', a)
        st = body[a + 2:b]
        js.append(f'___tmp = {st[1:]}; ___vrml += eval("___tmp");' if st.startswith('=') else st)
        pos = b + 2
    code = '\n'.join(js)
    imports = re.findall(r"import(.*?from.*?'.*?')[;\n]", code)
    code = re.sub(r"import(.*?from.*?'.*?')[;\n]", '', code)
    return imports, code


def render(proto, fields, tmp):
    imports, code = to_js(proto)
    with open(os.path.join(tmp, 'wbrandom.mjs'), 'w') as f:
        f.write(WBRANDOM)
    imp = '\n'.join('import' + i.replace("'wbrandom.js'", "'./wbrandom.mjs'") + ';' for i in imports)
    flds = ', '.join(f'{k}: {{value: {json.dumps(v)}}}' for k, v in fields.items())
    src = (f'{imp}\nconst stderr = []; const console = {{error: (...a) => stderr.push(a.join(""))}};\n'
           f'function render(t) {{ return t; }}\nlet ___vrml = ""; let ___tmp;\nconst fields = {{ {flds} }};\n'
           f'{code}\nprocess.stdout.write(JSON.stringify({{vrml: ___vrml, stderr}}));\n')
    path = os.path.join(tmp, 'tpl.mjs')
    with open(path, 'w') as f:
        f.write(src)
    out = json.loads(subprocess.run(['node', path], capture_output=True, text=True, check=True).stdout)
    return out['vrml'], out['stderr']


def barrels_of(vrml):
    t = re.findall(r'translation ([-\d.e]+) ([-\d.e]+) ([-\d.e]+)\s*rotation 0 0 1 ([-\d.e]+)\s*name "([^"]+)"', vrml)
    out = []
    for x, y, z, yaw, name in t:
        x, y, yaw = float(x), float(y), float(yaw)
        c, s = math.cos(yaw), math.sin(yaw)   # back to the collision cylinder centre, map frame
        out.append((x + c * P.BARREL_OFFSET[0] - s * P.BARREL_OFFSET[1] - P.WORLD_FROM_MAP[0],
                    y + s * P.BARREL_OFFSET[0] + c * P.BARREL_OFFSET[1] - P.WORLD_FROM_MAP[1], float(z), name))
    return out


def barriers_of(barrels):
    """Group barrels into barriers: two barrels whose cylinders touch belong together."""
    groups, left = [], list(range(len(barrels)))
    while left:
        g = [left.pop(0)]
        for k in list(left):
            if any(math.hypot(barrels[k][0] - barrels[i][0], barrels[k][1] - barrels[i][1]) <= 2 * P.BARREL_R + P.PAIR_GAP + 1e-3
                   for i in g):
                g.append(k)
                left.remove(k)
        groups.append(g)
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, default=20)
    args = ap.parse_args()
    proto = open(P.PROTO).read()
    g = P.compute(open(P.WORLD).read())
    ox, oy, res = g['ox'], g['oy'], g['res']
    at = lambda m, x, y: m[int((y - oy) / res), int((x - ox) / res)]  # noqa: E731
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        cases = [dict(randomSeed=s, task='extended', count=0, pPair=0.5) for s in range(1, args.seeds + 1)]
        cases += [dict(randomSeed=s, task='basic', count=0, pPair=0.5) for s in range(1, 6)]
        cases += [dict(randomSeed=7, task='extended', count=P.COUNTS['extended'][1], pPair=1.0)]
        first = None
        for f in cases:
            vrml, err = render(proto, f, tmp)
            b = barrels_of(vrml)
            groups = barriers_of(b)
            lo, hi = P.COUNTS[f['task']]
            n_ok = (f['count'] or lo) <= len(groups) <= (f['count'] or hi)
            sizes_ok = all(len(gr) <= 2 for gr in groups)
            gap = min((math.hypot(b[i][0] - b[j][0], b[i][1] - b[j][1]) - 2 * P.BARREL_R
                       for a_, ga in enumerate(groups) for gb in groups[a_ + 1:] for i in ga for j in gb),
                      default=math.inf)
            fence = min(at(g['d_fence'], x, y) for x, y, _, _ in b) - P.BARREL_R
            zone = min(at(g['d_zone'], x, y) for x, y, _, _ in b) - P.BARREL_R
            in_track = all(at(g['track'], x, y) for x, y, _, _ in b)
            names_ok = len({n for *_, n in b}) == len(b)
            ok = (n_ok and sizes_ok and gap >= P.GAP_BARRIERS - 1e-6 and fence >= P.GAP_FENCE - res
                  and zone >= P.GAP_ZONES - res and in_track and names_ok and not err)
            bad += not ok
            print(f"seed {f['randomSeed']:2d} {f['task']:8s} count {f['count']:2d}: {len(groups):2d} barriers "
                  f"({sum(len(gr) == 2 for gr in groups)} pairs), gap {gap:5.2f} (>= {P.GAP_BARRIERS:.2f}), "
                  f"fence {fence:4.2f} (>= {P.GAP_FENCE:.2f}), zones {zone:5.2f} (>= {P.GAP_ZONES:.2f}) "
                  f"{'ok' if ok else 'FAIL'} {err if err else ''}")
            if first is None:
                first = (f, b)
        # Webots seeds with Date.now() when randomSeed <= 0: two loads must differ
        a = barrels_of(render(proto, dict(randomSeed=0, task='extended', count=0, pPair=0.5), tmp)[0])
        import time
        time.sleep(0.01)
        c = barrels_of(render(proto, dict(randomSeed=0, task='extended', count=0, pPair=0.5), tmp)[0])
        print('randomSeed 0, two renders differ:', a != c)
    f, b = first
    img = np.dstack([np.where(g['occ'], 90, 255).astype(np.uint8)] * 3)
    img[g['allowed']] = (210, 250, 210)
    for x, y, _, _ in b:
        cv2.circle(img, (int((x - ox) / res), int((y - oy) / res)), 4, (0, 110, 255), -1)
    path = os.path.join(P.OUT, 'robocross_barrels_proto.png')
    cv2.imwrite(path, img[::-1])
    print(f"{'all cases pass' if not bad else f'{bad} cases FAIL'}; layout of seed {f['randomSeed']}: {path}")


if __name__ == '__main__':
    main()
