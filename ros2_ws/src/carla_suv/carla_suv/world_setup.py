"""Load the Robocross map into CARLA and scatter the barrels.

Map: the cooked Unreal map `Robocross` (carla/Import) if the server has it, else the OpenDRIVE
file made by tools/robocross_assets.py (--xodr). Barrels are a simplified version of the rules in
webots_ros2_suv/tools/place_barrels.py: random free cells, away from the fence and the spawn point.
The full rule set (gaps between barriers/zones, pairs) comes with the referee port.
"""
import argparse
import math
import os
import random
import sys
import time

import numpy as np
import yaml

WORLD_FROM_MAP = (2.2, -1.6)   # world = map + this, see webots_ros2_suv/maps/README.md
SPAWN = (2.435, -2.261)
SPAWN_SAFE = 15.0              # m around the spawn with no barrel
FENCE_CLEARANCE = 1.5          # m between a barrel and the nearest occupied cell
COUNTS = {'basic': (12, 18), 'extended': (18, 32)}
SIM_STEP = 0.05               # s per tick; the ticker runs at 1 / SIM_STEP Hz: real time
MIN_SPACING = 4.0              # m between barrels


def load_map(yaml_path):
    """Return (free-space mask [row 0 = north], resolution, origin_xy)."""
    import cv2
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    img = cv2.imread(os.path.join(os.path.dirname(yaml_path), meta['image']), cv2.IMREAD_GRAYSCALE)
    prob = (255 - img) / 255.0 if meta.get('negate', 0) == 0 else img / 255.0
    return prob < meta['free_thresh'], meta['resolution'], meta['origin'][:2]


def barrel_positions(free, res, origin, seed, task, count=None):
    """World (x, y) in ROS axes (y up), deterministic for a seed."""
    import cv2
    rng = random.Random(seed)
    clearance = int(FENCE_CLEARANCE / res)
    kernel = np.ones((2 * clearance + 1, 2 * clearance + 1), np.uint8)
    safe = cv2.erode(free.astype(np.uint8), kernel)
    height = free.shape[0]
    # The area outside the fence is "free" in the map too: keep only the track, i.e. the free
    # region connected to the spawn point.
    _, labels = cv2.connectedComponents(safe, connectivity=4)
    spawn_col = int((SPAWN[0] - WORLD_FROM_MAP[0] - origin[0]) / res)
    spawn_row = height - 1 - int((SPAWN[1] - WORLD_FROM_MAP[1] - origin[1]) / res)
    track = labels[spawn_row, spawn_col]
    if track == 0:
        raise ValueError('spawn point is not on free space of the map')
    rows, cols = np.nonzero(labels == track)

    lo, hi = COUNTS[task]
    wanted = count if count else rng.randint(lo, hi)
    picked = []
    for i in rng.sample(range(len(rows)), len(rows)):
        x = origin[0] + cols[i] * res + WORLD_FROM_MAP[0]
        y = origin[1] + (height - 1 - rows[i]) * res + WORLD_FROM_MAP[1]
        if math.hypot(x - SPAWN[0], y - SPAWN[1]) < SPAWN_SAFE:
            continue
        if any(math.hypot(x - px, y - py) < MIN_SPACING for px, py in picked):
            continue
        picked.append((x, y))
        if len(picked) == wanted:
            break
    return picked


def connect(host, port, timeout=240.0):
    import carla
    client = carla.Client(host, port)
    client.set_timeout(5.0)
    deadline = time.monotonic() + timeout
    while True:
        try:
            client.get_server_version()
            client.set_timeout(120.0)   # loading a map is slow
            return client
        except RuntimeError:
            if time.monotonic() > deadline:
                raise
            time.sleep(1.0)


def load_world(client, map_name, xodr):
    import carla
    available = [m.split('/')[-1] for m in client.get_available_maps()]
    if map_name in available:
        if client.get_world().get_map().name.split('/')[-1] != map_name:
            return client.load_world(map_name)
        return client.reload_world()   # same map: still drop the actors of a previous run
    if xodr:
        print(f'map {map_name} not on the server, using OpenDRIVE {xodr}', flush=True)
        with open(xodr) as f:
            params = carla.OpendriveGenerationParameters(
                vertex_distance=2.0, max_road_length=500.0, wall_height=0.0,
                additional_width=0.6, smooth_junctions=True,
                enable_mesh_visibility=True, enable_pedestrian_navigation=False)
            return client.generate_opendrive_world(f.read(), params)
    raise SystemExit(f'map {map_name} is not on the server and no --xodr given. Available: {available}')


def spawn_barrels(world, positions):
    import carla
    lib = world.get_blueprint_library()
    found = lib.filter('static.prop.barrel*') or lib.filter('static.prop.trafficcone01')
    bp = found[0]
    for old in world.get_actors().filter('static.prop.*'):
        if old.attributes.get('role_name') == 'robocross_barrel':
            old.destroy()
    bp.set_attribute('role_name', 'robocross_barrel') if bp.has_attribute('role_name') else None
    for x, y in positions:
        # ROS axes -> CARLA (left-handed): y flips.
        world.try_spawn_actor(bp, carla.Transform(carla.Location(x=x, y=-y, z=0.5)))
    print(f'spawned {len(positions)} x {bp.id}', flush=True)


def run_ticker(client, step=SIM_STEP, hold=0.0):
    """Hold the world in synchronous mode and tick it in real time.

    The server alone runs as fast as the GPU allows (10x real time on an empty map), which breaks
    every wall-clock controller. The bridge runs in passive mode and just follows these ticks.
    """
    world = client.get_world()
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = step
    world.apply_settings(settings)
    # Frozen for `hold` s: sim time does not advance while the sensors are spawned one after another,
    # so they share one sensor_tick phase (the stereo pair fires on the same frames). Spawning takes
    # milliseconds; the launch file starts it 5 s after the ticker.
    time.sleep(hold)
    print(f'ticking at {1 / step:.0f} Hz', flush=True)
    try:
        next_tick = time.monotonic()
        while True:
            world.tick()
            next_tick += step
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -1.0:
                next_tick = time.monotonic()   # fell behind (map load, paused): do not burst
    except KeyboardInterrupt:
        pass
    finally:
        try:
            world.apply_settings(original)   # a synchronous world nobody ticks is frozen
        except RuntimeError:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host', default=os.environ.get('CARLA_HOST', 'localhost'))
    ap.add_argument('--port', type=int, default=int(os.environ.get('CARLA_PORT', 2000)))
    ap.add_argument('--map', default='Robocross')
    ap.add_argument('--xodr', default='', help='OpenDRIVE file used if the map is not on the server')
    ap.add_argument('--map-yaml', default='', help='maps/robocross.yaml')
    ap.add_argument('--seed', type=int, default=0, help='barrel seed, <= 0 random')
    ap.add_argument('--task', choices=sorted(COUNTS), default='basic')
    ap.add_argument('--count', type=int, default=0)
    ap.add_argument('--no-barrels', action='store_true')
    ap.add_argument('--step', type=float, default=SIM_STEP, help='s per tick (the IMU cannot publish faster)')
    ap.add_argument('--hold', type=float, default=0.0, help='ticker: stay frozen this many s before ticking')
    ap.add_argument('--ticker', action='store_true', help='only tick the world in real time (runs until killed)')
    args = ap.parse_args(argv)

    client = connect(args.host, args.port)
    if args.ticker:
        return run_ticker(client, args.step, args.hold)
    world = load_world(client, args.map, args.xodr)
    if not args.no_barrels:
        seed = args.seed if args.seed > 0 else random.randrange(1, 2 ** 31)
        free, res, origin = load_map(args.map_yaml)
        spawn_barrels(world, barrel_positions(free, res, origin, seed, args.task, args.count))
        print(f'barrels: randomSeed {seed}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
