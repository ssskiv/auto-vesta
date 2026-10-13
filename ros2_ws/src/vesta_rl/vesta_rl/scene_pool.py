"""A fixed pool of training scenes, generated once and checked by a person.

A scene = a map (world + global path) and a layout of obstacles the global
planner does not know about (+ "ghost" lidar marks). The pool holds N maps of
all kinds (routes on the robocross map, procedural tracks with barriers on
both sides / one side / none, dashed or object rows) and L layouts per map, so
training sees N * L fixed combinations instead of an endless random stream.

Workflow (run inside the container, from /ros2_ws):

    ros2 run vesta_rl scene_pool generate --out src/vesta_rl/data/pool \\
        --map src/webots_ros2_suv/maps/robocross.yaml --paths src/vesta_rl/data/paths.npz
    # look at src/vesta_rl/data/pool/previews/*.png, or go through scenes one by one:
    ros2 run vesta_rl scene_pool review --pool src/vesta_rl/data/pool
    ros2 run vesta_rl train ... --pool src/vesta_rl/data/pool

Bad scenes go to <pool>/rejected.txt (the review window writes it, or edit it
by hand): one per line, "17" = the whole map 17, "17/5" = layout 5 of map 17,
"#" starts a comment. Training never uses rejected scenes.

Every scene is checked by a Nav2-like planner (feasibility.py: Hybrid-A* with
Dubins shots, turning radius 5 m, 1 m from every obstacle, within 4.5 m of the
global path). A scene without a route is thrown away and a new one is drawn;
a map that fails too often is replaced. The found route is stored with the
scene: the previews draw it and the BC expert drives along it. Training itself
does no checks.
"""
import argparse
import collections
import json
import math
import multiprocessing as mp
import os

import numpy as np

from . import config as C
from . import feasibility as F
from .obs import PathTracker
from .reward import CTE_LIMIT, GOAL_TOL
from .world import World, connected_components, cv2, distance_to_true

# map kinds and their default share of the pool: (name, procgen layout, procgen style)
KINDS = (
    ('robocross', None, None),
    ('two_sides/dashed', 'two_sides', 'dashed'),
    ('two_sides/objects', 'two_sides', 'objects'),
    ('two_sides/mixed', 'two_sides', None),
    ('one_side', 'one_side', None),
    ('open', 'open', None),
)
KIND_SHARE = (0.25, 0.15, 0.15, 0.15, 0.15, 0.15)

# obstacle rows: (kind, cx, cy, a, b, theta); box: a x b metres, disc: radius a
BOX, DISC = 0, 1
P_ON_PATH = 0.35          # share of obstacles put right on the path (forces a detour)
OBSTACLE_KINDS = ('box', 'barrel', 'wall')
OBSTACLE_P = (0.45, 0.35, 0.20)
P_GHOSTS = 0.3            # share of layouts with false lidar marks
MIN_PATH_FOR_OBSTACLES = 30.0

# quick necessary condition before the planner: base_link needs half the car
# width + the planner's margin to every obstacle (- a cell for rasterization)
CORRIDOR = F.CORRIDOR
CLEARANCE = C.FOOTPRINT_Y[1] + F.MARGIN - 0.2
LAYOUT_TRIES = 15         # random layouts drawn per accepted one before the map is replaced
MAP_TRIES = 20


# ---- obstacles -----------------------------------------------------------
def shape_cells(world, kind, cx, cy, a, b, th):
    """Grid cells (mx, my) covered by one obstacle row, inside the world."""
    res = world.res
    if kind == BOX:
        u = np.arange(-a / 2, a / 2 + 1e-9, res / 2)
        v = np.arange(-b / 2, b / 2 + 1e-9, res / 2)
        uu, vv = np.meshgrid(u, v)
        x = cx + math.cos(th) * uu - math.sin(th) * vv
        y = cy + math.sin(th) * uu + math.cos(th) * vv
    else:
        u = np.arange(-a, a + 1e-9, res / 2)
        uu, vv = np.meshgrid(u, u)
        inside = uu ** 2 + vv ** 2 <= a ** 2
        x, y = cx + uu[inside], cy + vv[inside]
    mx, my = world.to_cell(x.ravel(), y.ravel())
    ok = world.inside(mx, my)
    if not ok.any():
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    cells = np.unique(np.stack([mx[ok], my[ok]]), axis=1)
    return cells[0], cells[1]


def ghost_cells(world, ghosts):
    """Cells of the ghost discs (rows (cx, cy, r))."""
    out = [shape_cells(world, DISC, cx, cy, r, 0.0, 0.0) for cx, cy, r in ghosts]
    if not out:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate([c[0] for c in out]), np.concatenate([c[1] for c in out])


def reachable(world, occ, tracker):
    """True if a car-wide free corridor connects the start of the path to its end
    without leaving CORRIDOR metres from the path (4-connected flood fill on the
    free cells at least CLEARANCE from every obstacle)."""
    w = world
    res = w.res
    pad = CORRIDOR + CLEARANCE + 1.0
    lo = tracker.xy.min(axis=0) - pad
    hi = tracker.xy.max(axis=0) + pad
    (x0, y0), (x1, y1) = w.to_cell(lo[0], lo[1]), w.to_cell(hi[0], hi[1])
    x0, y0, x1, y1 = int(x0), int(y0), int(x1) + 1, int(y1) + 1
    win = np.zeros((y1 - y0, x1 - x0), dtype=bool)  # outside the map = open ground
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, w.shape[1]), min(y1, w.shape[0])
    if sx1 > sx0 and sy1 > sy0:
        win[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = occ[sy0:sy1, sx0:sx1]

    xy = tracker.xy
    dense = np.concatenate([np.linspace(a, b, 4, endpoint=False) for a, b in zip(xy[:-1], xy[1:])] + [xy[-1:]])
    pmx, pmy = w.to_cell(dense[:, 0], dense[:, 1])
    on_path = np.zeros_like(win)
    on_path[pmy - y0, pmx - x0] = True
    region = ((distance_to_true(on_path) * res <= CORRIDOR)
              & (distance_to_true(win) * res >= CLEARANCE)).astype(np.uint8)

    (smx, smy), (gmx, gmy) = w.to_cell(xy[0, 0], xy[0, 1]), w.to_cell(xy[-1, 0], xy[-1, 1])
    start, goal = (int(smy) - y0, int(smx) - x0), (int(gmy) - y0, int(gmx) - x0)
    if not (region[start] and region[goal]):
        return False
    labels = connected_components(region)
    return labels[start] == labels[goal]


def random_layout(rng, tracker, max_obstacles=15, p_on_path=P_ON_PATH):
    """One random layout: (obstacles (K, 6), ghosts (G, 3)), not checked."""
    rows = []
    if tracker.length >= MIN_PATH_FOR_OBSTACLES and max_obstacles > 0:
        for _ in range(rng.integers(1, max_obstacles + 1)):
            s = rng.uniform(12.0, tracker.length - 8.0)
            x, y = tracker.point_at(s)
            nx, ny = tracker.point_at(s + 0.5)
            heading = math.atan2(ny - y, nx - x)
            off = rng.uniform(-1.0, 1.0) if rng.random() < p_on_path else rng.uniform(-6.0, 6.0)
            cx, cy = x - math.sin(heading) * off, y + math.cos(heading) * off
            kind = rng.choice(OBSTACLE_KINDS, p=OBSTACLE_P)
            if kind == 'box':
                rows.append((BOX, cx, cy, rng.uniform(0.4, 2.5), rng.uniform(0.4, 2.5), rng.uniform(-math.pi, math.pi)))
            elif kind == 'barrel':
                rows.append((DISC, cx, cy, rng.uniform(0.25, 0.6), 0.0, 0.0))
            else:  # wall segment, mostly across the path
                rows.append((BOX, cx, cy, rng.uniform(2.0, 6.0), rng.uniform(0.2, 0.4),
                             heading + math.pi / 2 + rng.normal(0.0, 0.5)))
    ghosts = []
    if rng.random() < P_GHOSTS:
        for _ in range(rng.integers(1, 6)):
            x, y = tracker.point_at(rng.uniform(5.0, tracker.length))
            ghosts.append((x + rng.uniform(-3, 3), y + rng.uniform(-3, 3), rng.uniform(0.2, 0.5)))
    return np.array(rows, np.float32).reshape(-1, 6), np.array(ghosts, np.float32).reshape(-1, 3)


def blocked_grid(world, obstacles, ghosts):
    """Static map + obstacles + ghosts (the arbiter stops for ghosts too)."""
    occ = world.occupied.copy()
    for row in obstacles:
        mx, my = shape_cells(world, int(row[0]), *row[1:])
        occ[my, mx] = True
    gx, gy = ghost_cells(world, ghosts)
    occ[gy, gx] = True
    return occ


def check_scene(world, tracker, planner, obstacles, ghosts):
    """The planner's route for the scene, or None."""
    occ = blocked_grid(world, obstacles, ghosts)
    if not reachable(world, occ, tracker):  # fast rejection of the obviously blocked ones
        return None
    return planner.check(occ)


# ---- pool on disk --------------------------------------------------------
def _save_world(path, world):
    np.savez_compressed(path, occupied=np.packbits(world.occupied), shape=np.array(world.shape),
                        origin=np.array([world.ox, world.oy]), res=world.res)


def _load_world(path):
    d = np.load(path)
    shape = tuple(d['shape'])
    occ = np.unpackbits(d['occupied'], count=shape[0] * shape[1]).reshape(shape).astype(bool)
    return World.from_grid(occ, float(d['origin'][0]), float(d['origin'][1]), float(d['res']))


def read_rejected(root):
    """(set of map ids, {map id: set of layout ids}) from <root>/rejected.txt."""
    maps, layouts = set(), collections.defaultdict(set)
    path = os.path.join(root, 'rejected.txt')
    if os.path.exists(path):
        for line in open(path):
            item = line.split('#')[0].strip()
            if not item:
                continue
            if '/' in item:
                i, j = item.split('/')
                layouts[int(i)].add(int(j))
            else:
                maps.add(int(item))
    return maps, layouts


def write_rejected(root, maps, layouts):
    lines = ['# rejected scenes: "map" or "map/layout", see scene_pool.py']
    lines += [str(i) for i in sorted(maps)]
    lines += [f'{i}/{j}' for i in sorted(layouts) for j in sorted(layouts[i]) if i not in maps]
    with open(os.path.join(root, 'rejected.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


class ScenePool:
    """Read side of a pool directory. Worlds of procedural maps are loaded on
    demand and a few are kept in memory."""

    CACHE = 12

    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, 'meta.json')) as f:
            self.meta = json.load(f)
        self.maps = []
        for i in range(self.meta['n_maps']):
            d = np.load(self._map_file(i))
            self.maps.append(dict(kind=str(d['kind']), world=str(d['world']), path=d['path'],
                                  obstacles=d['obstacles'], obs_off=d['obs_off'],
                                  ghosts=d['ghosts'], ghost_off=d['ghost_off'],
                                  routes=d['routes'], route_off=d['route_off']))
        self._worlds = collections.OrderedDict()
        self.reload_rejected()

    def _map_file(self, i):
        return os.path.join(self.root, 'maps', f'm{i:03d}.npz')

    def reload_rejected(self):
        self.rejected_maps, self.rejected_layouts = read_rejected(self.root)
        self.choices = []
        for i, m in enumerate(self.maps):
            if i in self.rejected_maps:
                continue
            ok = [j for j in range(self.n_layouts(i)) if j not in self.rejected_layouts.get(i, ())]
            if ok:
                self.choices.append((i, np.array(ok)))
        if not self.choices:
            raise RuntimeError(f'{self.root}: every scene is rejected')
        self.kind_maps = collections.defaultdict(list)  # kind -> indices into choices
        for c, (i, _) in enumerate(self.choices):
            self.kind_maps[self.kind(i)].append(c)
        self.kind_weights = None

    def set_kind_weights(self, weights):
        """{kind: weight} (None = every non-rejected map equally likely)."""
        if weights is None:
            self.kind_weights = None
            return
        kinds = [k for k in self.kind_maps if weights.get(k, 0.0) > 0.0]
        w = np.array([weights[k] for k in kinds], float)
        self.kind_weights = (kinds, w / w.sum())

    def __len__(self):
        return len(self.maps)

    def n_layouts(self, i):
        return len(self.maps[i]['obs_off']) - 1

    def n_scenes(self):
        return sum(len(js) for _, js in self.choices)

    def kind(self, i):
        return self.maps[i]['kind']

    def path(self, i):
        return self.maps[i]['path']

    def world(self, i):
        name = self.maps[i]['world']
        key = f'm{i:03d}' if name == 'own' else name
        if key not in self._worlds:
            file = self._map_file(i) if name == 'own' else os.path.join(self.root, 'worlds', name + '.npz')
            self._worlds[key] = _load_world(file)
            while len(self._worlds) > self.CACHE:
                self._worlds.popitem(last=False)
        self._worlds.move_to_end(key)
        return self._worlds[key]

    def layout(self, i, j):
        m = self.maps[i]
        return (m['obstacles'][m['obs_off'][j]:m['obs_off'][j + 1]],
                m['ghosts'][m['ghost_off'][j]:m['ghost_off'][j + 1]])

    def route(self, i, j):
        """The planner's route (n, 2) for scene (i, j), every feasibility.ROUTE_STEP metres."""
        m = self.maps[i]
        return m['routes'][m['route_off'][j]:m['route_off'][j + 1]]

    def sample(self, rng):
        """A random non-rejected (map, layout): maps equally likely (or a kind by its
        weight, then its maps equally likely), then layouts."""
        if self.kind_weights is None:
            i, js = self.choices[rng.integers(len(self.choices))]
        else:
            kinds, w = self.kind_weights
            cs = self.kind_maps[kinds[rng.choice(len(kinds), p=w)]]
            i, js = self.choices[cs[rng.integers(len(cs))]]
        return i, int(js[rng.integers(len(js))])


# ---- generation ----------------------------------------------------------
_G = {}


def _init_worker(map_yaml, paths, nice):
    from .paths import load_paths
    if nice:
        os.nice(nice)
    _G['robocross'] = World(map_yaml)
    _G['library'] = [p for p in load_paths(paths) if PathTracker(p).length >= MIN_PATH_FOR_OBSTACLES]


def _make_map(task):
    """Generates map i with its layouts and writes <out>/maps/mXXX.npz. Draws a
    new map (procedural, or another route on robocross) when the empty map has
    no route or too few random layouts get one."""
    from . import procgen
    i, kind, layout, style, seed, args = task
    rng = np.random.default_rng(seed)
    n_layouts = args['layouts']
    tried = 0
    for map_try in range(MAP_TRIES):
        if kind == 'robocross':
            world, own = _G['robocross'], False
            path = _G['library'][rng.integers(len(_G['library']))]
        else:
            world, path = procgen.generate(rng, layout=layout, style=style)
            own = True
        tracker = PathTracker(path)
        planner = F.Feasibility(world, tracker)
        if planner.check(world.occupied) is None:
            continue
        obstacles, ghosts, routes = [], [], []
        for _ in range(n_layouts * LAYOUT_TRIES):
            tried += 1
            o, g = random_layout(rng, tracker, args['max_obstacles'])
            route = check_scene(world, tracker, planner, o, g)
            if route is not None:
                obstacles.append(o)
                ghosts.append(g)
                routes.append(route.astype(np.float32))
                if len(routes) == n_layouts:
                    break
        if len(routes) == n_layouts:
            break
    else:
        raise RuntimeError(f'map {i} ({kind}): no drivable map in {MAP_TRIES} tries')

    out = dict(kind=kind, world='own' if own else 'robocross', path=path.astype(np.float64),
               obstacles=np.concatenate(obstacles), obs_off=np.cumsum([0] + [len(o) for o in obstacles]),
               ghosts=np.concatenate(ghosts), ghost_off=np.cumsum([0] + [len(g) for g in ghosts]),
               routes=np.concatenate(routes), route_off=np.cumsum([0] + [len(r) for r in routes]))
    if own:
        out.update(occupied=np.packbits(world.occupied), shape=np.array(world.shape),
                   origin=np.array([world.ox, world.oy]), res=world.res)
    np.savez_compressed(os.path.join(args['out'], 'maps', f'm{i:03d}.npz'), **out)
    return dict(i=i, kind=kind, length=tracker.length, maps_tried=map_try + 1, layouts_tried=tried,
                mean_obstacles=float(np.mean([len(o) for o in obstacles])))


def generate(args):
    out = args.out
    if os.path.exists(os.path.join(out, 'meta.json')) and not args.overwrite:
        raise SystemExit(f'{out} already has a pool; pass --overwrite to replace it')
    if os.path.exists(os.path.join(out, 'rejected.txt')):  # its ids refer to the old scenes
        os.replace(os.path.join(out, 'rejected.txt'), os.path.join(out, 'rejected.old.txt'))
    os.makedirs(os.path.join(out, 'maps'), exist_ok=True)
    os.makedirs(os.path.join(out, 'worlds'), exist_ok=True)
    for f in os.listdir(os.path.join(out, 'maps')):
        os.remove(os.path.join(out, 'maps', f))

    rng = np.random.default_rng(args.seed)
    share = np.array(KIND_SHARE, float)
    if args.robocross_share is not None:
        share[0] = args.robocross_share
        share[1:] *= (1.0 - args.robocross_share) / share[1:].sum()
    counts = np.floor(share * args.maps).astype(int)
    counts[np.argsort(-(share * args.maps - counts))[:args.maps - counts.sum()]] += 1

    tasks, i = [], 0
    worker_args = dict(out=out, layouts=args.layouts, max_obstacles=args.max_obstacles)
    for (kind, layout, style), n in zip(KINDS, counts):
        for _ in range(n):
            tasks.append((i, kind, layout, style, int(rng.integers(2 ** 63)), worker_args))
            i += 1

    _save_world(os.path.join(out, 'worlds', 'robocross.npz'), World(args.map))
    print(f'generating {len(tasks)} maps x {args.layouts} layouts on {args.workers} workers '
          f'({", ".join(f"{k[0]} {n}" for k, n in zip(KINDS, counts))})', flush=True)
    stats = []
    with mp.get_context('spawn').Pool(args.workers, _init_worker, (args.map, args.paths, args.nice)) as p:
        for done, st in enumerate(p.imap_unordered(_make_map, tasks), 1):
            stats.append(st)
            print(f'[{done}/{len(tasks)}] map {st["i"]:3d} {st["kind"]:18s} path {st["length"]:5.0f} m, '
                  f'maps tried {st["maps_tried"]}, layouts accepted {args.layouts}/{st["layouts_tried"]}, '
                  f'{st["mean_obstacles"]:.1f} obstacles per layout', flush=True)
    stats.sort(key=lambda st: st['i'])
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump(dict(n_maps=len(tasks), layouts=args.layouts, max_obstacles=args.max_obstacles, seed=args.seed,
                       map_yaml=args.map, paths=args.paths,
                       planner=dict(turn_radius=F.TURN_RADIUS, margin=F.MARGIN, corridor=F.CORRIDOR),
                       stats=stats), f, indent=1)
    by_kind = collections.defaultdict(lambda: [0, 0, 0])
    for st in stats:
        k = by_kind[st['kind']]
        k[0] += args.layouts
        k[1] += st['layouts_tried']
        k[2] += st['maps_tried'] - 1
    for kind, (acc, tried, maps_rej) in by_kind.items():
        print(f'{kind:18s} layouts accepted {acc / tried:4.0%}, maps replaced {maps_rej}')
    write_rejected(out, set(), {})
    print(f'pool: {len(tasks)} maps, {len(tasks) * args.layouts} scenes in {out}')
    if args.sheets > 0:
        previews(ScenePool(out), pages=range(args.sheets))


# ---- pictures for the review ---------------------------------------------
COLORS = dict(free=(15, 23, 42), corridor=(30, 41, 66), map=(158, 168, 184), obstacle=(245, 158, 51),
              ghost=(192, 132, 252), path=(96, 165, 250), car=(74, 222, 128), goal=(248, 113, 113),
              text=(229, 231, 235), rejected=(239, 68, 68))
VIEW_MARGIN = 8.0  # m around the path


def render(pool, i, j, size=480):
    """RGB image (size x size, north up) of scene (i, j): static map, obstacles,
    ghosts, the 5 m band around the path where the car may drive, the car at the
    start to scale and the goal tolerance circle."""
    world, path = pool.world(i), pool.path(i)
    obstacles, ghosts = pool.layout(i, j)
    lo = path.min(axis=0) - VIEW_MARGIN
    hi = path.max(axis=0) + VIEW_MARGIN
    side = float(max(hi - lo))
    lo = (lo + hi) / 2 - side / 2
    n = int(math.ceil(side / world.res))
    (x0, y0) = world.to_cell(lo[0], lo[1])
    x0, y0 = int(x0), int(y0)

    def crop(full):
        out = np.zeros((n, n), dtype=bool)
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x0 + n, world.shape[1]), min(y0 + n, world.shape[0])
        if sx1 > sx0 and sy1 > sy0:
            out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = full[sy0:sy1, sx0:sx1]
        return out

    def cells_mask(mx, my):
        out = np.zeros((n, n), dtype=bool)
        mx, my = mx - x0, my - y0
        ok = (mx >= 0) & (mx < n) & (my >= 0) & (my < n)
        out[my[ok], mx[ok]] = True
        return out

    static = crop(world.occupied)
    obst = np.zeros((n, n), dtype=bool)
    for row in obstacles:
        obst |= cells_mask(*shape_cells(world, int(row[0]), *row[1:]))
    ghost = cells_mask(*ghost_cells(world, ghosts))
    dense = np.concatenate([np.linspace(a, b, 4, endpoint=False) for a, b in zip(path[:-1], path[1:])] + [path[-1:]])
    on_path = cells_mask(*world.to_cell(dense[:, 0], dense[:, 1]))
    band = distance_to_true(on_path) * world.res <= CTE_LIMIT

    def small(mask):  # any cell set -> pixel set (thin obstacles stay visible)
        return cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA) > 1e-3

    img = np.empty((size, size, 3), np.uint8)
    img[:] = COLORS['free']
    for mask, color in ((band, 'corridor'), (static, 'map'), (obst, 'obstacle'), (ghost, 'ghost')):
        img[small(mask)] = COLORS[color]
    img = np.ascontiguousarray(img[::-1])

    k = size / side
    to_px = lambda xy: np.stack([(xy[..., 0] - lo[0]) * k, size - (xy[..., 1] - lo[1]) * k], -1)  # noqa: E731
    cv2.polylines(img, [np.round(to_px(path)).astype(np.int32)], False, COLORS['path'], 1, cv2.LINE_AA)
    cv2.polylines(img, [np.round(to_px(pool.route(i, j))).astype(np.int32)], False, COLORS['car'], 1, cv2.LINE_AA)
    yaw = math.atan2(*(path[min(2, len(path) - 1)] - path[0])[::-1])
    c, s = math.cos(yaw), math.sin(yaw)
    fx = np.array([C.FOOTPRINT_X[0], C.FOOTPRINT_X[1], C.FOOTPRINT_X[1], C.FOOTPRINT_X[0]])
    fy = np.array([C.FOOTPRINT_Y[0], C.FOOTPRINT_Y[0], C.FOOTPRINT_Y[1], C.FOOTPRINT_Y[1]])
    car = np.stack([path[0, 0] + c * fx - s * fy, path[0, 1] + s * fx + c * fy], -1)
    cv2.fillPoly(img, [np.round(to_px(car)).astype(np.int32)], COLORS['car'], cv2.LINE_AA)
    g = np.round(to_px(path[-1])).astype(int)
    cv2.circle(img, (int(g[0]), int(g[1])), max(2, int(round(GOAL_TOL * k))), COLORS['goal'], 1, cv2.LINE_AA)

    # 10 m scale bar
    bar = int(round(10.0 * k))
    cv2.line(img, (8, size - 10), (8 + bar, size - 10), COLORS['text'], 2)
    cv2.putText(img, '10 m', (12 + bar, size - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text'], 1, cv2.LINE_AA)
    rejected = i in pool.rejected_maps or j in pool.rejected_layouts.get(i, ())
    label = f'{i}/{j}  {pool.kind(i)}  {len(obstacles)} obst' + ('  REJECTED' if rejected else '')
    cv2.putText(img, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                COLORS['rejected' if rejected else 'text'], 1, cv2.LINE_AA)
    if rejected:
        cv2.rectangle(img, (0, 0), (size - 1, size - 1), COLORS['rejected'], 3)
    return img


SHEET_COLS, SHEET_ROWS = 4, 4


def sheet(pool, i, page, tile=480):
    """Layouts page*16 .. page*16+15 of map i on one image."""
    per = SHEET_COLS * SHEET_ROWS
    js = range(page * per, min((page + 1) * per, pool.n_layouts(i)))
    gap = 4
    img = np.zeros((SHEET_ROWS * (tile + gap), SHEET_COLS * (tile + gap), 3), np.uint8)
    img[:] = (51, 65, 85)
    for k, j in enumerate(js):
        r, c = divmod(k, SHEET_COLS)
        img[r * (tile + gap):r * (tile + gap) + tile, c * (tile + gap):c * (tile + gap) + tile] = render(pool, i, j, tile)
    return img


def previews(pool, maps=None, pages=(0,)):
    out = os.path.join(pool.root, 'previews')
    os.makedirs(out, exist_ok=True)
    per = SHEET_COLS * SHEET_ROWS
    for i in (range(len(pool)) if maps is None else maps):
        for page in pages:
            if page * per >= pool.n_layouts(i):
                continue
            file = os.path.join(out, f'm{i:03d}_p{page}.png')
            cv2.imwrite(file, cv2.cvtColor(sheet(pool, i, page), cv2.COLOR_RGB2BGR))
        print('previews of map', i, flush=True)
    legend = ('previews: one sheet per map and page, 16 layouts each, labelled "map/layout". '
              'Green = the car at the start (to scale) and the route the planner found, blue line = global path, '
              'red circle = goal, '
              'dark blue band = 5 m from the path (further = episode over), grey = static map, '
              'orange = obstacles the planner does not know, violet = ghost lidar marks.\n')
    with open(os.path.join(out, 'README.txt'), 'w') as f:
        f.write(legend)


# ---- interactive review ---------------------------------------------------
REVIEW_KEYS = ('right / space: next layout   left: previous   up / down: previous / next map\n'
               'x: reject / restore this layout   m: reject / restore the whole map   q: quit')


def review(pool, start_map=0, start_layout=0):
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    state = dict(i=start_map, j=start_layout)
    fig, ax = plt.subplots(figsize=(10, 10.6), facecolor='#0f172a')
    fig.canvas.manager.set_window_title('vesta_rl scene review')
    im = ax.imshow(render(pool, state['i'], state['j'], 900))
    ax.set_axis_off()
    fig.text(0.01, 0.005, REVIEW_KEYS, family='monospace', fontsize=9, va='bottom')
    title = fig.suptitle('', fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    def show():
        i, j = state['i'], state['j']
        im.set_data(render(pool, i, j, 900))
        n_rej = len(pool.rejected_maps) + sum(len(v) for v in pool.rejected_layouts.values())
        title.set_text(f'map {i + 1}/{len(pool)} ({pool.kind(i)})   layout {j + 1}/{pool.n_layouts(i)}   '
                       f'rejected entries: {n_rej}   usable scenes: {pool.n_scenes()}')
        fig.canvas.draw_idle()

    def save():
        write_rejected(pool.root, pool.rejected_maps, pool.rejected_layouts)
        try:
            pool.reload_rejected()
        except RuntimeError as e:
            print(e)

    def on_key(e):
        i, j = state['i'], state['j']
        if e.key in ('right', ' '):
            if j + 1 < pool.n_layouts(i):
                state['j'] = j + 1
            elif i + 1 < len(pool):
                state['i'], state['j'] = i + 1, 0
        elif e.key == 'left':
            if j > 0:
                state['j'] = j - 1
            elif i > 0:
                state['i'], state['j'] = i - 1, pool.n_layouts(i - 1) - 1
        elif e.key == 'down':
            state['i'], state['j'] = min(len(pool) - 1, i + 1), 0
        elif e.key == 'up':
            state['i'], state['j'] = max(0, i - 1), 0
        elif e.key == 'x':
            pool.rejected_layouts[i] ^= {j}
            save()
        elif e.key == 'm':
            pool.rejected_maps ^= {i}
            save()
        elif e.key == 'q':
            plt.close(fig)
            return
        show()

    fig.canvas.mpl_connect('key_press_event', on_key)
    show()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('generate', help='generate a new pool')
    g.add_argument('--out', required=True)
    g.add_argument('--map', required=True, help='robocross.yaml')
    g.add_argument('--paths', required=True, help='path library on that map (make_paths)')
    g.add_argument('--maps', type=int, default=100)
    g.add_argument('--layouts', type=int, default=200, help='obstacle layouts per map')
    g.add_argument('--max-obstacles', type=int, default=15)
    g.add_argument('--robocross-share', type=float, default=None,
                   help=f'share of maps that are routes on the robocross map (default {KIND_SHARE[0]})')
    g.add_argument('--sheets', type=int, default=1, help='preview pages (16 layouts each) per map to draw')
    g.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    g.add_argument('--nice', type=int, default=10)
    g.add_argument('--seed', type=int, default=0)
    g.add_argument('--overwrite', action='store_true')
    p = sub.add_parser('preview', help='draw preview sheets')
    p.add_argument('--pool', required=True)
    p.add_argument('--maps', type=int, nargs='*', help='map ids (default: all)')
    p.add_argument('--pages', type=int, nargs='*', default=[0], help='pages of 16 layouts')
    r = sub.add_parser('review', help='go through scenes one by one and reject bad ones')
    r.add_argument('--pool', required=True)
    r.add_argument('--map', type=int, default=0)
    r.add_argument('--layout', type=int, default=0)
    args = ap.parse_args()

    if args.cmd == 'generate':
        generate(args)
    elif args.cmd == 'preview':
        previews(ScenePool(args.pool), args.maps, args.pages)
    else:
        review(ScenePool(args.pool), args.map, args.layout)


if __name__ == '__main__':
    main()
