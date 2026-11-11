#!/usr/bin/env python3
"""Rearrange the barrels of robocross.wbt by the RoboCross.Trassa rules.

Rules: "Регламент РобоКросс.Трасса-2024" (version 17.12.2023), russianrobotics.ru:
  8.6.1   a barrier is one barrel or two barrels touching side by side, 0.4-2.0 m wide;
  18.25   extended task: 9 to 16 barriers (18.24, basic task: 6 to 9 -> --task basic);
  18.27   between the nearest barriers at least L + W of the vehicle in any direction
          (also >= L along and >= 2 W across the track, both implied by L + W);
  18.28   at least 2 W from the track fence;
  18.29   at least L from the start-finish zone and the intermediate zone;
  8.3.1   barriers stand in the main and the additional zone.
L, W: the Lada Vesta boundingObject of LadaVesta.proto (4.555 x 1.73 m).

Zones on this map (there are no zone lines in the world; the Winder gates stand in for them):
  start-finish: the top left corner behind the WinderStart and the WinderFinish gate lines;
  intermediate: the quadrilateral between the WinderFreeDeal and the WinderGaz gates;
  main + additional zone: the rest of the corridor connected to the vehicle spawn.
Fence: the occupied cells of maps/robocross.pgm.

Default: writes protos/RobocrossBarrels.proto and puts `DEF BARRELS RobocrossBarrels {}`
into worlds/robocross.wbt in place of the PlasticBarrel nodes. The PROTO's JavaScript template
places the barrels every time Webots loads the world (randomSeed <= 0: a new layout each time,
> 0: a fixed one). New layout during a run, through vesta_driver (the robot is the supervisor):
    ros2 topic pub --once /vesta/barrels_seed std_msgs/msg/Int32 "{data: 7}"
It removes DEF BARRELS and imports it again with the new randomSeed. Checked in Webots R2025a
(vesta-sim image, Xvfb): the template expands without errors, the layout equals the node
render of tools/check_barrels_proto.py. Do NOT change randomSeed with setSFInt32 on the live
node: Webots segfaulted when a controller had read the barrels inside the PROTO and then
changed the field; remove + import with reads in between ran without problems.
Webots templates cannot read files, so the map is baked into the PROTO as a bit mask of the
cells where a barrel centre may stand (fence, zones and track already applied); the template
only draws the barriers and keeps them apart. Rerun this script after changing the map.

    python3 tools/place_barrels.py                 # PROTO mode
    python3 tools/place_barrels.py --static [--seed 0] [--count 12] [--task basic]
        # fixed PlasticBarrel nodes written into the world, preview tools/robocross_barrels.png

The old world goes to tools/robocross.wbt.bak. The installed world is a copy: rebuild the
package (colcon build --packages-select webots_ros2_suv) before launching.
tools/check_barrels_proto.py renders the PROTO outside Webots and checks the rules.
"""
import argparse
import math
import os
import re
import shutil

import cv2
import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLD = os.path.join(PKG, 'worlds', 'robocross.wbt')
MAP_YAML = os.path.join(PKG, 'maps', 'robocross.yaml')
OUT = os.path.dirname(os.path.abspath(__file__))   # tools/ is not installed by setup.py, worlds/ is

VEH_L, VEH_W = 4.555, 1.73          # LadaVesta.proto boundingObject: x -0.8775 .. 3.6775, y +-0.865
BARREL_R = 0.45                      # PlasticBarrel.proto boundingObject: Cylinder radius 0.45 ...
BARREL_OFFSET = (-0.05, -0.05)       # ... at translation -0.05 -0.05 0.75 in the barrel frame
BARREL_Z = 0.1                       # height of the barrels already in the world
GAP_BARRIERS = VEH_L + VEH_W         # 18.27
GAP_FENCE = 2 * VEH_W                # 18.28
GAP_ZONES = VEH_L                    # 18.29
# Two barrels of a pair stand this far apart, not exactly touching: ODE's cylinder-cylinder
# contact of two parallel cylinders at exactly 2 R gives NaN, and a NaN body breaks the
# collision space for every body (the car and all barrels then fall through the ground).
PAIR_GAP = 0.02
SPAWN_SAFE = 15.0                    # m around the vehicle spawn with no barrel
RULE_COUNTS = {'basic': (6, 9), 'extended': (9, 16)}   # 18.24 / 18.25
DENSITY = 2                          # x the rule counts: more barriers than the regulation, on request
COUNTS = {k: (lo * DENSITY, hi * DENSITY) for k, (lo, hi) in RULE_COUNTS.items()}
# Webots world = map + this. The map frame is odom (the first GPS fix), not the world; fitted
# so that the 74 Barrier blocks of the world land on occupied map cells (57 of 74 within a cell).
WORLD_FROM_MAP = (2.2, -1.6)

PROTO = os.path.join(PKG, 'protos', 'RobocrossBarrels.proto')
MASK_RES = 0.4      # 2 x 2 map cells; a mask cell is allowed only if all four are


def write_proto(allowed, ox, oy, res, patches, path=PROTO):
    f = int(round(MASK_RES / res))
    H, W = (allowed.shape[0] // f) * f, (allowed.shape[1] // f) * f
    mask = allowed[:H, :W].reshape(H // f, f, W // f, f).all(axis=(1, 3))
    bits = np.packbits(mask.ravel(), bitorder='big')
    hexmask = bits.tobytes().hex()
    x0, y0 = ox + WORLD_FROM_MAP[0], oy + WORLD_FROM_MAP[1]
    js_patches = ', '.join(f'[{cx}, {cy}, {yaw}, {sx}, {sy}]' for cx, cy, yaw, sx, sy in patches)
    text = PROTO_TEMPLATE
    for k, v in dict(MASK=hexmask, MW=mask.shape[1], MH=mask.shape[0], MRES=MASK_RES, X0=x0, Y0=y0,
                     PATCHES=js_patches, R=BARREL_R, PG=PAIR_GAP, GAP=GAP_BARRIERS, OFFX=BARREL_OFFSET[0],
                     OFFY=BARREL_OFFSET[1], Z=BARREL_Z, LO_B=COUNTS['basic'][0], HI_B=COUNTS['basic'][1],
                     LO_E=COUNTS['extended'][0], HI_E=COUNTS['extended'][1], DENSITY=DENSITY).items():
        text = text.replace('@' + k + '@', str(v))
    with open(path, 'w') as out:
        out.write(text)
    print(f'{path}: mask {mask.shape[1]} x {mask.shape[0]} cells of {MASK_RES} m, '
          f'{int(mask.sum())} allowed ({mask.mean():.1%})')
    return mask


def use_proto_in_world(world, wbt):
    node = re.compile(r'^PlasticBarrel \{\n.*?^\}\n', re.S | re.M)
    gen = re.compile(r'^DEF BARRELS RobocrossBarrels \{.*?^\}\n', re.S | re.M)
    first = node.search(wbt) or gen.search(wbt)
    assert first, 'neither PlasticBarrel nor BARRELS in the world'
    n_old = len(node.findall(wbt))
    keep = gen.search(wbt)                                   # keep the fields someone set on it
    block = keep.group(0) if keep else 'DEF BARRELS RobocrossBarrels {\n}\n'
    body = wbt[:first.start()] + block + gen.sub('', node.sub('', wbt[first.start():]))
    # IMPORTABLE: vesta_driver re-imports the node to change the layout; it also declares it for the world
    ext = 'IMPORTABLE EXTERNPROTO "../protos/RobocrossBarrels.proto"\n'
    body = re.sub(r'^(IMPORTABLE )?EXTERNPROTO "\.\./protos/RobocrossBarrels\.proto"\n', '', body, flags=re.M)
    if ext not in body:
        anchor = 'EXTERNPROTO "../protos/PlasticBarrel.proto"\n'
        assert anchor in body
        body = body.replace(anchor, anchor + ext)
    backup = os.path.join(OUT, os.path.basename(world) + '.bak')
    shutil.copyfile(world, backup)
    with open(world, 'w') as out:
        out.write(body)
    print(f'{world}: {n_old} PlasticBarrel nodes replaced by DEF BARRELS RobocrossBarrels (old world: {backup})')


# Webots evaluates %< code >% and %<= expression >% as JavaScript; plain text goes into a
# template literal, so it must not contain backticks or "${".
PROTO_TEMPLATE = r"""#VRML_SIM R2023b utf8
# template language: javascript
# Barrels of robocross.wbt placed by the RoboCross.Trassa-2024 rules (see webots_ros2_suv/tools/place_barrels.py).
# New layout during a run: remove DEF BARRELS and import it with another randomSeed (vesta_driver,
# topic /vesta/barrels_seed). setSFInt32 on the live node crashed Webots R2025a (place_barrels.py).
# GENERATED by tools/place_barrels.py from maps/robocross.pgm: do not edit, rerun the script.

EXTERNPROTO "PlasticBarrel.proto"

PROTO RobocrossBarrels [
  field SFInt32  randomSeed 0           # <= 0: a new layout at every load, > 0: a fixed layout
  field SFString task       "extended"  # "basic": @LO_B@-@HI_B@ barriers, "extended": @LO_E@-@HI_E@ (18.24/18.25 x @DENSITY@)
  field SFInt32  count      0           # > 0: exactly this many barriers
  field SFFloat  pPair      0.5         # share of barriers made of two barrels touching (8.6.1)
]
{
%<
  import * as wbrandom from 'wbrandom.js';
  if (fields.randomSeed.value <= 0)
    wbrandom.seed(Date.now());
  else
    wbrandom.seed(fields.randomSeed.value);

  // allowed barrel centres, world frame: bit (row j, column i) of the hex string, row 0 = min y
  const MASK = '@MASK@';
  const MW = @MW@, MH = @MH@, MRES = @MRES@, X0 = @X0@, Y0 = @Y0@;
  const R = @R@, PG = @PG@, GAP = @GAP@, OFFX = @OFFX@, OFFY = @OFFY@, Z = @Z@;
  const PATCHES = [@PATCHES@];   // UnevenTerrain (x, y, yaw, size x, size y): barrels drop onto the bumps

  function allowed(x, y) {
    const i = Math.floor((x - X0) / MRES), j = Math.floor((y - Y0) / MRES);
    if (i < 0 || j < 0 || i >= MW || j >= MH)
      return false;
    const k = j * MW + i;
    return ((parseInt(MASK[k >> 2], 16) >> (3 - (k & 3))) & 1) === 1;
  }

  const range = fields.task.value === 'basic' ? [@LO_B@, @HI_B@] : [@LO_E@, @HI_E@];
  const count = fields.count.value > 0 ? fields.count.value : wbrandom.integer(range[0], range[1]);
  const barriers = [];
  for (let t = 0; t < 200000 && barriers.length < count; ++t) {
    const x = X0 + wbrandom.real(0, MW * MRES), y = Y0 + wbrandom.real(0, MH * MRES);
    const c = [[x, y]];
    if (wbrandom.real(0, 1) < fields.pPair.value) {
      const th = wbrandom.real(0, Math.PI);
      c.push([x + (2 * R + PG) * Math.cos(th), y + (2 * R + PG) * Math.sin(th)]);   // PG: see place_barrels.py
    }
    if (!c.every(p => allowed(p[0], p[1])))
      continue;
    let ok = true;
    for (const b of barriers)
      for (const p of b)
        for (const q of c)
          if (Math.hypot(p[0] - q[0], p[1] - q[1]) - 2 * R < GAP)   // 18.27: edge to edge >= L + W
            ok = false;
    if (ok)
      barriers.push(c);
  }
  if (barriers.length < count)
    console.error('RobocrossBarrels: only ' + barriers.length + ' of ' + count + ' barriers fit the rules');

  const barrels = [];
  for (const b of barriers)
    for (const p of b) {
      const yaw = wbrandom.real(-Math.PI, Math.PI);
      const cs = Math.cos(yaw), sn = Math.sin(yaw);
      const x = p[0] - (cs * OFFX - sn * OFFY), y = p[1] - (sn * OFFX + cs * OFFY);   // cylinder on the centre
      let z = Z;
      for (const q of PATCHES) {
        const u = Math.cos(q[2]) * (x - q[0]) + Math.sin(q[2]) * (y - q[1]);
        const v = -Math.sin(q[2]) * (x - q[0]) + Math.cos(q[2]) * (y - q[1]);
        if (Math.abs(u) <= q[3] / 2 && Math.abs(v) <= q[4] / 2)
          z = Z + 0.5;   // UnevenTerrain is up to 0.5 m high
      }
      barrels.push([x, y, z, yaw]);
    }
>%
  Group {
    children [
%< for (let k = 0; k < barrels.length; ++k) { >%
      PlasticBarrel {
        translation %<= barrels[k][0] >% %<= barrels[k][1] >% %<= barrels[k][2] >%
        rotation 0 0 1 %<= barrels[k][3] >%
        name %<= '"barrel ' + k + '"' >%
      }
%< } >%
    ]
  }
}
"""


def read_map(yaml_path):
    meta = dict(line.split(':', 1) for line in open(yaml_path) if ':' in line)
    image = os.path.join(os.path.dirname(yaml_path), meta['image'].strip())
    res = float(meta['resolution'])
    origin = [float(v) for v in meta['origin'].strip().strip('[]').split(',')]
    data = open(image, 'rb').read()
    tokens, pos = [], 0
    while len(tokens) < 4:                       # P5 header: magic width height maxval, # comments
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b'#':
            pos = data.index(b'\n', pos) + 1
            continue
        start = pos
        while not data[pos:pos + 1].isspace():
            pos += 1
        tokens.append(data[start:pos])
    assert tokens[0] == b'P5'
    w, h = int(tokens[1]), int(tokens[2])
    pix = np.frombuffer(data, np.uint8, w * h, pos + 1).reshape(h, w)[::-1]   # row 0 = min y
    occupied = (255.0 - pix) / 255.0 > float(meta['occupied_thresh'])
    return occupied, origin[0], origin[1], res


def gates(wbt, proto):
    pts = re.findall(proto + r' \{\s*translation ([-\d.e]+) ([-\d.e]+)', wbt)
    assert len(pts) == 2, f'{proto}: expected 2 posts, found {len(pts)}'
    return [(float(x) - WORLD_FROM_MAP[0], float(y) - WORLD_FROM_MAP[1]) for x, y in pts]


def uneven_patches(wbt):
    """(cx, cy, yaw, sx, sy) of the UnevenTerrain nodes, world frame."""
    out = []
    for block in re.findall(r'UnevenTerrain \{(.*?)\n\}', wbt, re.S):
        t = re.search(r'translation ([-\d.e]+) ([-\d.e]+)', block)
        r = re.search(r'rotation [-\d.e]+ [-\d.e]+ ([-\d.e]+) ([-\d.e]+)', block)
        s = re.search(r'size ([-\d.e]+) ([-\d.e]+)', block)
        yaw = float(r[2]) * (1 if float(r[1]) >= 0 else -1) if r else 0.0
        out.append((float(t[1]), float(t[2]), yaw, float(s[1]), float(s[2])))
    return out


def side(p, a, b):
    """> 0 left of the line a -> b."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def zone_masks(X, Y, wbt, spawn):
    start, finish = gates(wbt, 'WinderStart'), gates(wbt, 'WinderFinish')
    s_start = np.sign(side(spawn, *start))
    s_finish = np.sign(side(spawn, *finish))
    # start-finish: on the spawn side of both gate lines, within the corner they close off
    corner = (np.sign(side((X, Y), *start)) == s_start) & (np.sign(side((X, Y), *finish)) == s_finish)
    corner &= X <= max(p[0] for p in start + finish) + 1.0
    corner &= Y >= min(p[1] for p in start + finish) - 1.0
    a, b = gates(wbt, 'WinderFreeDeal')
    c, d = gates(wbt, 'WinderGaz')
    quad = np.array([a, b, d, c] if np.hypot(b[0] - d[0], b[1] - d[1]) < np.hypot(b[0] - c[0], b[1] - c[1])
                    else [a, b, c, d])
    inter = np.ones_like(corner)
    for k in range(4):                           # convex quadrilateral: same side of every edge
        p, q = quad[k], quad[(k + 1) % 4]
        cen = quad.mean(0)
        inter &= np.sign(side((X, Y), p, q)) == np.sign(side(cen, p, q))
    return corner, inter, quad


def dist_m(mask, res):
    """Distance (m) from every cell to the nearest True cell."""
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) * res


def compute(wbt, spawn_safe=SPAWN_SAFE):
    """Map, zones and the cells where a barrel centre may stand (map frame, 0.2 m grid)."""
    occ, ox, oy, res = read_map(MAP_YAML)
    H, W = occ.shape
    X, Y = np.meshgrid(ox + (np.arange(W) + 0.5) * res, oy + (np.arange(H) + 0.5) * res)
    v = re.search(r'DEF Vesta1 LadaVesta \{\s*translation ([-\d.e]+) ([-\d.e]+)', wbt)
    spawn = (float(v[1]) - WORLD_FROM_MAP[0], float(v[2]) - WORLD_FROM_MAP[1])
    corner, inter, quad = zone_masks(X, Y, wbt, spawn)
    # the track: free cells connected to the spawn once the fence clearance is taken off
    # (closes the gaps between barrier blocks and trees, so the outside does not count)
    d_fence = dist_m(occ, res)
    sx, sy = int((spawn[0] - ox) / res), int((spawn[1] - oy) / res)
    reach = (d_fence >= GAP_FENCE).astype(np.uint8)
    _, labels = cv2.connectedComponents(reach, connectivity=4)
    assert reach[sy, sx], 'the spawn is too close to the fence'
    track = labels == labels[sy, sx]
    d_zone = dist_m(corner | inter, res)
    allowed = track & (d_fence >= GAP_FENCE + BARREL_R) & (d_zone >= GAP_ZONES + BARREL_R) & ~corner & ~inter
    allowed &= np.hypot(X - spawn[0], Y - spawn[1]) >= spawn_safe + BARREL_R   # safe zone around the spawn
    return dict(occ=occ, ox=ox, oy=oy, res=res, X=X, Y=Y, corner=corner, inter=inter, track=track,
                allowed=allowed, d_fence=d_fence, d_zone=d_zone)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=None, help='default: random')
    ap.add_argument('--task', choices=COUNTS, default='extended')
    ap.add_argument('--count', type=int, help='number of barriers (default: random in the task range)')
    ap.add_argument('--p-pair', type=float, default=0.5, help='share of two-barrel barriers')
    ap.add_argument('--world', default=WORLD)
    ap.add_argument('--static', action='store_true', help='write fixed PlasticBarrel nodes instead of the PROTO')
    ap.add_argument('--dry-run', action='store_true', help='do not touch the world')
    ap.add_argument('--spawn-safe', type=float, default=SPAWN_SAFE, help='m around the vehicle spawn with no barrel')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    lo, hi = COUNTS[args.task]
    count = args.count if args.count is not None else int(rng.integers(lo, hi + 1))
    if not lo <= count <= hi:
        print(f'warning: {count} barriers is outside the {args.task} range {lo}-{hi} (18.24/18.25 x {DENSITY})')

    wbt = open(args.world).read()
    g = compute(wbt, args.spawn_safe)
    occ, ox, oy, res, X, Y = g['occ'], g['ox'], g['oy'], g['res'], g['X'], g['Y']
    corner, inter, track, allowed = g['corner'], g['inter'], g['track'], g['allowed']
    d_fence, d_zone, H, W = g['d_fence'], g['d_zone'], occ.shape[0], occ.shape[1]

    if not args.static:
        write_proto(allowed, ox, oy, res, uneven_patches(wbt))
        if not args.dry_run:
            use_proto_in_world(args.world, wbt)
        return

    def ok_at(x, y):
        i, j = int((y - oy) / res), int((x - ox) / res)
        return 0 <= i < H and 0 <= j < W and allowed[i, j]

    cells = np.argwhere(allowed)
    barriers = []                                # list of barrel centre lists
    for _ in range(200000):
        if len(barriers) == count:
            break
        i, j = cells[rng.integers(len(cells))]
        cx, cy = X[i, j] + rng.uniform(-res / 2, res / 2), Y[i, j] + rng.uniform(-res / 2, res / 2)
        centres = [(cx, cy)]
        if rng.random() < args.p_pair:           # 8.6.1: two barrels touching side by side
            th = rng.uniform(0, math.pi)
            centres.append((cx + (2 * BARREL_R + PAIR_GAP) * math.cos(th), cy + (2 * BARREL_R + PAIR_GAP) * math.sin(th)))
        if not all(ok_at(*c) for c in centres):
            continue
        gap = min((math.hypot(a[0] - b[0], a[1] - b[1]) - 2 * BARREL_R
                   for other in barriers for a in centres for b in other), default=math.inf)
        if gap >= GAP_BARRIERS:
            barriers.append(centres)
    if len(barriers) < count:
        raise SystemExit(f'only {len(barriers)} of {count} barriers fit the rules; try another --seed or fewer')

    # self-check of every rule on the result
    barrels = [c for b in barriers for c in b]
    min_gap = min(math.hypot(a[0] - b[0], a[1] - b[1]) - 2 * BARREL_R
                  for k, p in enumerate(barriers) for q in barriers[k + 1:] for a in p for b in q)
    at = lambda m, c: m[int((c[1] - oy) / res), int((c[0] - ox) / res)]  # noqa: E731
    fence = min(at(d_fence, c) for c in barrels) - BARREL_R
    zone = min(at(d_zone, c) for c in barrels) - BARREL_R
    assert min_gap >= GAP_BARRIERS and fence >= GAP_FENCE - res and zone >= GAP_ZONES - res
    print(f'{len(barriers)} barriers ({sum(len(b) == 2 for b in barriers)} pairs), {len(barrels)} barrels')
    print(f'nearest barriers {min_gap:.2f} m apart (>= {GAP_BARRIERS:.2f}), fence {fence:.2f} m '
          f'(>= {GAP_FENCE:.2f}), zones {zone:.2f} m (>= {GAP_ZONES:.2f}); distances from the 0.2 m map grid')

    # preview
    img = np.dstack([np.where(occ, 90, 255).astype(np.uint8)] * 3)
    img[track & ~allowed & ~occ] = (235, 235, 200)
    img[allowed] = (210, 250, 210)
    img[corner & ~occ] = (180, 180, 255)
    img[inter & ~occ] = (255, 200, 160)
    px = lambda x, y: (int((x - ox) / res), int((y - oy) / res))  # noqa: E731
    for b in barriers:
        for c in b:
            cv2.circle(img, px(*c), max(2, int(BARREL_R / res)) + 2, (0, 110, 255), -1)
    preview = os.path.join(OUT, 'robocross_barrels.png')
    out = img[::-1].copy()
    for k, (t, col) in enumerate([('red: start-finish zone', (120, 120, 255)), ('blue: intermediate zone', (255, 160, 100)),
                                  ('green: allowed for barrel centres', (0, 170, 0)), ('orange: barrels', (0, 110, 255))]):
        cv2.putText(out, t, (10, 25 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.imwrite(preview, out)
    print('preview', preview)
    if args.dry_run:
        return

    # world: drop every PlasticBarrel node, put the new ones where the first one was
    node = re.compile(r'^PlasticBarrel \{\n.*?^\}\n', re.S | re.M)
    first = node.search(wbt)
    assert first, 'no PlasticBarrel in the world'
    patches = uneven_patches(wbt)
    lines = []
    for k, (mx, my) in enumerate(barrels):
        x, y = mx + WORLD_FROM_MAP[0], my + WORLD_FROM_MAP[1]
        yaw = rng.uniform(-math.pi, math.pi)
        c, s = math.cos(yaw), math.sin(yaw)
        x -= c * BARREL_OFFSET[0] - s * BARREL_OFFSET[1]     # put the collision cylinder on the centre
        y -= s * BARREL_OFFSET[0] + c * BARREL_OFFSET[1]
        on_bumps = any(abs(math.cos(t) * (x - px_) + math.sin(t) * (y - py_)) <= sx_ / 2
                       and abs(-math.sin(t) * (x - px_) + math.cos(t) * (y - py_)) <= sy_ / 2
                       for px_, py_, t, sx_, sy_ in patches)
        z = BARREL_Z + (0.5 if on_bumps else 0.0)            # UnevenTerrain is up to 0.5 m high: drop from above
        lines.append(f'PlasticBarrel {{\n  translation {x:.4f} {y:.4f} {z}\n  rotation 0 0 1 {yaw:.4f}\n'
                     f'  name "PlasticBarrel({k})"\n}}\n')
    rest = node.sub('', wbt[first.start():])
    new = wbt[:first.start()] + ''.join(lines) + rest
    backup = os.path.join(OUT, os.path.basename(args.world) + '.bak')
    shutil.copyfile(args.world, backup)
    with open(args.world, 'w') as f:
        f.write(new)
    print(f'{args.world}: {len(node.findall(wbt))} barrels replaced by {len(barrels)} (old world: {backup})')


if __name__ == '__main__':
    main()
