"""Blender: build the Robocross terrain meshes and write one FBX for CARLA's `make import`.

    blender -b -P robocross_scene.py -- <robocross.wbt> <assets dir with fence.obj> <out.fbx>

Everything comes from the Webots world, so the CARLA map stays the same track:
  * Terrain_Sand      the RectangleArena floor (500 x 500 m)
  * Road_Road_Asphalt the Road nodes (20 m wide, B-spline through the way points)
  * Terrain_Mud_N     the UnevenTerrain patches (noise bumps, flat edges)
  * Terrain_Fence     the fence made by robocross_assets.py (CARLA axes in the obj)
Meshes are authored in Webots axes (x forward, y left, z up). The Unreal FBX import mirrors Y, so
CARLA y = -y of the file, which is what the world uses. UVs: one unit = one texture tile (4 m sand/asphalt, 5 m mud, 2 m fence).
Units: meters; CARLA's import applies the x100 itself (global_scale stays 1).
"""
import math
import os
import random
import re
import sys

import bmesh
import bpy
from bpy_extras.io_utils import axis_conversion


def parse_world(path):
    """The few nodes we need, as plain dicts. A tiny reader: the .wbt fields are one per line."""
    text = open(path).read()

    def block(name):
        for m in re.finditer(r'^\s*(?:DEF \w+ )?%s \{' % name, text, re.M):
            depth, i = 0, m.end() - 1
            while True:
                depth += {'{': 1, '}': -1}.get(text[i], 0)
                if depth == 0:
                    break
                i += 1
            yield text[m.start():i + 1]

    def vec(body, field, default):
        m = re.search(r'^\s*%s ([^\n]+)' % field, body, re.M)
        return [float(v) for v in m.group(1).split()] if m else default

    arena = next(block('RectangleArena'))
    world = {'arena_center': vec(arena, 'translation', [0, 0, 0])[:2],
             'arena_size': vec(arena, 'floorSize', [500, 500]), 'roads': [], 'uneven': []}

    m = re.search(r'DEF Road Solid \{\s*translation ([^\n]+)', text)
    world['road_offset'] = [float(v) for v in m.group(1).split()] if m else [0, 0, 0]
    for body in block('Road'):
        pts = re.search(r'wayPoints \[(.*?)\]', body, re.S)
        if pts:
            world['roads'].append({
                'width': vec(body, 'width', [7])[0],
                'points': [[float(v) for v in ln.split()] for ln in pts.group(1).strip().splitlines()]})
    for body in block('UnevenTerrain'):
        world['uneven'].append({
            'translation': vec(body, 'translation', [0, 0, 0]),
            'angle': (vec(body, 'rotation', [0, 0, 1, 0])[3] if 'rotation 0 0 1' in body else 0.0),
            'size': vec(body, 'size', [5, 5, 1]),
            'dims': (int(vec(body, 'xDimension', [10])[0]), int(vec(body, 'yDimension', [10])[0])),
            'seed': int(vec(body, 'randomSeed', [1])[0]),
            'octaves': int(vec(body, 'perlinNOctaves', [3])[0])})
    return world


def make_object(name, verts, faces, uvs):
    """faces: list of vertex-index tuples. uvs: per vertex (u, v)."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    layer = mesh.uv_layers.new(name='UVMap')
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            layer.data[li].uv = uvs[mesh.loops[li].vertex_index]
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def grid(nx, ny, x0, y0, sx, sy, height=lambda x, y: 0.0, uv_scale=1.0):
    verts, uvs, faces = [], [], []
    for j in range(ny + 1):
        for i in range(nx + 1):
            x, y = x0 + sx * i / nx, y0 + sy * j / ny
            verts.append((x, y, height(x, y)))
            uvs.append((x * uv_scale, y * uv_scale))
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            faces.append((a, a + 1, a + nx + 2, a + nx + 1))   # counter-clockwise seen from +z
    return verts, faces, uvs


def bspline(points, subdivision=4):
    """Open uniform cubic B-spline like Webots' Road (splineSubdivision): through the end points."""
    p = [tuple(q[:2]) for q in points]
    if len(p) < 3:
        return p
    padded = [p[0]] * 2 + p + [p[-1]] * 2
    out = []
    for k in range(len(padded) - 3):
        a, b, c, d = padded[k:k + 4]
        for s in range(subdivision):
            t = s / subdivision
            w = ((1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6,
                 (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6)
            out.append((sum(w[i] * q[0] for i, q in enumerate((a, b, c, d))),
                        sum(w[i] * q[1] for i, q in enumerate((a, b, c, d)))))
    out.append(p[-1])
    return out


def road_mesh(world):
    verts, faces, uvs = [], [], []
    ox, oy, oz = world['road_offset']
    travelled = 0.0
    for road in world['roads']:
        line = bspline(road['points'])
        half = road['width'] / 2
        base = len(verts)
        prev = None
        for k, (x, y) in enumerate(line):
            nxt = line[min(k + 1, len(line) - 1)]
            before = line[max(k - 1, 0)]
            dx, dy = nxt[0] - before[0], nxt[1] - before[1]
            norm = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / norm, dx / norm
            if prev is not None:
                travelled += math.hypot(x - prev[0], y - prev[1])
            prev = (x, y)
            for side in (+1, -1):
                verts.append((x + ox + side * half * nx, y + oy + side * half * ny, oz))
                uvs.append(((side * half + half) * 0.25, travelled * 0.25))
        for k in range(len(line) - 1):
            a = base + 2 * k
            faces.append((a, a + 1, a + 3, a + 2))
    return make_object('Road_Road_Asphalt', verts, faces, uvs)


def value_noise(seed, octaves):
    rng = random.Random(seed)
    tables = [[[rng.random() for _ in range(34)] for _ in range(34)] for _ in range(octaves)]

    def smooth(t):
        return t * t * (3 - 2 * t)

    def sample(table, x, y):
        i, j = int(x) % 33, int(y) % 33
        fx, fy = smooth(x - int(x)), smooth(y - int(y))
        a = table[j][i] * (1 - fx) + table[j][i + 1] * fx
        b = table[j + 1][i] * (1 - fx) + table[j + 1][i + 1] * fx
        return a * (1 - fy) + b * fy

    def noise(u, v):    # u, v in 0..1
        total, amp, norm = 0.0, 1.0, 0.0
        for o, table in enumerate(tables):
            total += amp * sample(table, u * 4 * 2 ** o % 32, v * 4 * 2 ** o % 32)
            norm += amp
            amp *= 0.5
        return total / norm
    return noise


def uneven_meshes(world):
    objs = []
    for n, patch in enumerate(world['uneven'], 1):
        sx, sy, sz = patch['size']
        nx, ny = patch['dims']
        noise = value_noise(patch['seed'], patch['octaves'])

        def height(x, y, sx=sx, sy=sy, sz=sz, noise=noise):
            u, v = x / sx + 0.5, y / sy + 0.5
            edge = min(u, 1 - u, v, 1 - v) * 2          # 0 at the border: flatBounds
            return sz * noise(u, v) * min(1.0, edge * 4)
        verts, faces, uvs = grid(nx - 1, ny - 1, -sx / 2, -sy / 2, sx, sy, height, uv_scale=0.2)
        a = patch['angle']
        tx, ty, tz = patch['translation']
        verts = [(tx + x * math.cos(a) - y * math.sin(a), ty + x * math.sin(a) + y * math.cos(a), tz + z)
                 for x, y, z in verts]
        objs.append(make_object('Terrain_Mud_%d' % n, verts, faces, uvs))
    return objs


def fence_mesh(assets):
    """fence.obj is in CARLA axes (y flipped): mirror it back into Webots axes, fix the winding."""
    bpy.ops.wm.obj_import(filepath=os.path.join(assets, 'fence.obj'), forward_axis='Y', up_axis='Z')
    obj = bpy.context.selected_objects[0]
    obj.name = obj.data.name = 'Terrain_Fence'
    obj.scale = (1.0, -1.0, 1.0)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.flip_normals()
    bpy.ops.object.mode_set(mode='OBJECT')
    # UVs for a repeating texture along the wall: u = distance along the segment, v = height.
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    layer = bm.loops.layers.uv.verify()
    for face in bm.faces:
        origin = face.verts[0].co
        for loop in face.loops:
            d = loop.vert.co - origin
            loop[layer].uv = (math.hypot(d.x, d.y) * 0.5, d.z * 0.5)
    bm.to_mesh(mesh)
    bm.free()
    return obj


def main():
    wbt, assets, out = sys.argv[sys.argv.index('--') + 1:][:3]
    world = parse_world(wbt)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    cx, cy = world['arena_center']
    w, h = world['arena_size']
    # 10 m cells: enough for a flat floor, the bumps are separate meshes
    verts, faces, uvs = grid(int(w // 10), int(h // 10), cx - w / 2, cy - h / 2, w, h, uv_scale=0.25)
    make_object('Terrain_Sand', verts, faces, uvs)
    road_mesh(world)
    uneven_meshes(world)
    fence_mesh(assets)

    for obj in bpy.data.objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

    from io_scene_fbx import export_fbx_bin

    class Op:
        def report(self, kind, msg):
            print(kind, msg)

    export_fbx_bin.save(
        Op(), bpy.context, filepath=out, global_scale=1.0, apply_scale_options='FBX_SCALE_NONE',
        axis_up='Z', axis_forward='-Y', global_matrix=axis_conversion(to_forward='-Y', to_up='Z').to_4x4(),
        object_types={'MESH'}, mesh_smooth_type='FACE', bake_space_transform=False)
    print('wrote', out, [(o.name, len(o.data.polygons)) for o in bpy.data.objects])


main()
