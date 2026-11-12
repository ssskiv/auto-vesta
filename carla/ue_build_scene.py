"""Unreal Python (run inside the editor / as a commandlet): dress the imported Robocross map.

    UE4Editor CarlaUE4.uproject -run=pythonscript -script=/path/ue_build_scene.py

Run after `make import` and before `make package`. It
  * makes materials from CARLA's stock textures (sand floor, mud patches, concrete fence) and assigns them
    to the imported meshes; the road gets CARLA's stock asphalt textures;
  * plants the Webots `Forest` nodes with CARLA's stock trees (pines, oaks, maples);
  * saves the map.
World = Webots coordinates, Unreal is in cm and y is mirrored (CARLA y = -ros y).
"""
import math
import random
import re

import unreal

WBT = '/home/dev/auto-vesta/ros2_ws/src/webots_ros2_suv/worlds/robocross.wbt'
MAP = '/Game/robocross/Maps/Robocross/Robocross'
MATERIAL_DIR = '/Game/robocross/Materials'
STOCK = '/Game/Carla/Static'
TEXTURES = STOCK + '/GenericMaterials/Ground/Textures'
TREES = STOCK + '/Vegetation/Trees/'
TREE_MIX = (['SM_Pine_L_v%d' % i for i in range(1, 9)] * 3 +
            ['SM_Pine_M_v%d' % i for i in range(1, 9)] * 3 +
            ['SM_Oak_M_v%d' % i for i in range(1, 5)] +
            ['SM_Maple_M_v%d' % i for i in range(1, 5)] + ['SM_Platanus01', 'SM_Platanus02'])
TREE_SPACING_M = 3.5     # between tree centers
FOREST_DENSITY_SCALE = 0.5   # Webots density is for a sparse look; shrink: StaticMeshActor count matters here

lib = unreal.MaterialEditingLibrary
assets = unreal.EditorAssetLibrary


def load(path):
    asset = assets.load_asset(path)
    if asset is None:
        raise RuntimeError('missing asset ' + path)
    return asset


def texture_material(name, base, normal=None, orm=None, tint=None, roughness=None):
    """Tiling is in the mesh UVs (one unit = one tile)."""
    path = '%s/%s' % (MATERIAL_DIR, name)
    if assets.does_asset_exist(path):
        assets.delete_asset(path)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    mat = tools.create_asset(name, MATERIAL_DIR, unreal.Material, unreal.MaterialFactoryNew())
    tc = lib.create_material_expression(mat, unreal.MaterialExpressionTextureCoordinate, -900, 0)

    def sampler(tex, y, kind=None):
        node = lib.create_material_expression(mat, unreal.MaterialExpressionTextureSample, -500, y)
        node.set_editor_property('texture', load(tex))
        if kind:
            node.set_editor_property('sampler_type', kind)
        lib.connect_material_expressions(tc, '', node, 'UVs')
        return node

    color = sampler(base, -300, unreal.MaterialSamplerType.SAMPLERTYPE_COLOR)
    if tint:
        mul = lib.create_material_expression(mat, unreal.MaterialExpressionMultiply, -250, -300)
        const = lib.create_material_expression(mat, unreal.MaterialExpressionConstant3Vector, -500, -500)
        const.set_editor_property('constant', unreal.LinearColor(*tint))
        lib.connect_material_expressions(color, 'RGB', mul, 'A')
        lib.connect_material_expressions(const, '', mul, 'B')
        lib.connect_material_property(mul, '', unreal.MaterialProperty.MP_BASE_COLOR)
    else:
        lib.connect_material_property(color, 'RGB', unreal.MaterialProperty.MP_BASE_COLOR)
    if normal:
        n = sampler(normal, 0, unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL)
        lib.connect_material_property(n, 'RGB', unreal.MaterialProperty.MP_NORMAL)
    if orm:
        srgb = load(orm).get_editor_property('srgb')   # the sampler type must match the texture flag
        o = sampler(orm, 300, unreal.MaterialSamplerType.SAMPLERTYPE_COLOR if srgb
                    else unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
        lib.connect_material_property(o, 'G', unreal.MaterialProperty.MP_ROUGHNESS)
        lib.connect_material_property(o, 'R', unreal.MaterialProperty.MP_AMBIENT_OCCLUSION)
    elif roughness is not None:
        r = lib.create_material_expression(mat, unreal.MaterialExpressionConstant, -250, 300)
        r.set_editor_property('r', roughness)
        lib.connect_material_property(r, '', unreal.MaterialProperty.MP_ROUGHNESS)
    lib.recompile_material(mat)
    assets.save_loaded_asset(mat)
    return mat


def build_materials():
    sand = texture_material('M_RobocrossSand', TEXTURES + '/BeachSand/T_SandBeach_D',
                            TEXTURES + '/BeachSand/T_SandBeach_N', TEXTURES + '/BeachSand/T_SandBeach_ORM')
    mud = texture_material('M_RobocrossMud', TEXTURES + '/Dirt/T_Dirt_d', TEXTURES + '/Dirt/T_Dirt_n',
                           tint=(0.62, 0.55, 0.5), roughness=0.95)
    fence = texture_material('M_RobocrossFence', TEXTURES + '/Dirt/T_Rock_Basalt_D',
                             TEXTURES + '/Dirt/T_Rock_Basalt_N', roughness=0.9)
    asphalt_dir = STOCK + '/GenericMaterials/Asphalt/Textures/Asphalt08/'
    asphalt = texture_material('M_RobocrossAsphalt', asphalt_dir + 'T_Asphalt08_d', asphalt_dir + 'T_Asphalt08_n',
                               asphalt_dir + 'T_Asphalt08_ORMH')
    return {'Terrain_Sand': sand, 'Terrain_Mud': mud, 'Terrain_Fence': fence, 'Road_Road_Asphalt': asphalt}


def dress_meshes(materials):
    for actor in unreal.EditorLevelLibrary.get_all_level_actors():
        if not isinstance(actor, unreal.StaticMeshActor):
            continue
        comp = actor.static_mesh_component
        mesh = comp.static_mesh
        if mesh is None:
            continue
        name = mesh.get_name()
        for key, mat in materials.items():
            if key in name:
                for slot in range(comp.get_num_materials()):
                    comp.set_material(slot, mat)
                    mesh.set_material(slot, mat)
                assets.save_loaded_asset(mesh)
                unreal.log('material %s -> %s' % (mat.get_name(), name))
                break
        else:
            unreal.log('left alone: ' + name)


def parse_forests():
    text = open(WBT).read()
    forests = []
    for m in re.finditer(r'^Forest \{(.*?)^\}', text, re.M | re.S):
        body = m.group(1)

        def field(name, default):
            f = re.search(r'^\s*%s ([^\n]+)' % name, body, re.M)
            return [float(v) for v in f.group(1).split()] if f else default
        pts = re.search(r'shape \[(.*?)\]', body, re.S)
        flat = [float(v) for v in pts.group(1).replace('\n', ' ').split()]
        tx, ty = field('translation', [0, 0, 0])[:2]
        forests.append({'shape': [(tx + flat[i], ty + flat[i + 1]) for i in range(0, len(flat), 2)],
                        'density': field('density', [0.2])[0], 'seed': int(field('randomSeed', [0])[0])})
    return forests


def inside(poly, x, y):
    hit = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def plant_forests():
    trees = [load(TREES + n) for n in TREE_MIX]
    planted = 0
    for forest in parse_forests():
        rng = random.Random(forest['seed'])
        xs = [p[0] for p in forest['shape']]
        ys = [p[1] for p in forest['shape']]
        area = 0.5 * abs(sum(forest['shape'][i][0] * forest['shape'][(i + 1) % len(forest['shape'])][1] -
                             forest['shape'][(i + 1) % len(forest['shape'])][0] * forest['shape'][i][1]
                             for i in range(len(forest['shape']))))
        want = int(area * forest['density'] * FOREST_DENSITY_SCALE)
        taken = []
        for _ in range(want * 20):
            if len(taken) >= want:
                break
            x, y = rng.uniform(min(xs), max(xs)), rng.uniform(min(ys), max(ys))
            if not inside(forest['shape'], x, y):
                continue
            if any(math.hypot(x - a, y - b) < TREE_SPACING_M for a, b in taken):
                continue
            taken.append((x, y))
            actor = unreal.EditorLevelLibrary.spawn_actor_from_object(
                rng.choice(trees), unreal.Vector(x * 100.0, -y * 100.0, 0.0),
                unreal.Rotator(0.0, rng.uniform(0, 360), 0.0))
            scale = rng.uniform(0.8, 1.15)
            actor.set_actor_scale3d(unreal.Vector(scale, scale, scale))
            actor.set_actor_label('Tree_%d' % planted)
            planted += 1
        unreal.log('forest area %.0f m2: %d trees' % (area, len(taken)))
    return planted


def main():
    unreal.EditorLevelLibrary.load_level(MAP)
    for actor in unreal.EditorLevelLibrary.get_all_level_actors():
        if actor.get_actor_label().startswith('Tree_'):
            actor.destroy_actor()          # idempotent: rerun replaces the forest
    dress_meshes(build_materials())
    unreal.log('planted %d trees' % plant_forests())
    unreal.EditorLevelLibrary.save_current_level()
    unreal.log('ue_build_scene: done')


main()
