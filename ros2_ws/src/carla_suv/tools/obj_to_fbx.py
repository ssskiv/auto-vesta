"""Blender: ground.obj + fence.obj (meters, CARLA axes) -> one FBX for CARLA's `make import`.

    blender -b -P obj_to_fbx.py -- carla/assets Robocross.fbx

Object names carry the CARLA mesh tags (Road_Road_*, Terrain_*). Blender is right-handed and the FBX
importer of Unreal mirrors Y, so Y is negated here: CARLA y of a point stays the same after import.
1 Blender unit = 1 m; the FBX header carries the unit, Unreal imports meters as 100 units (verified with get_level_bbs).
"""
import os
import sys

import bpy

src, out = sys.argv[sys.argv.index('--') + 1:][:2]

bpy.ops.wm.read_factory_settings(use_empty=True)
for filename, name in (('ground.obj', 'Road_Road_Ground'), ('fence.obj', 'Terrain_Fence')):
    bpy.ops.wm.obj_import(filepath=os.path.join(src, filename), forward_axis='Y', up_axis='Z')
    obj = bpy.context.selected_objects[0]
    obj.name = name
    obj.data.name = name

for obj in bpy.data.objects:
    obj.scale = (1.0, -1.0, 1.0)
bpy.ops.object.select_all(action='SELECT')
bpy.context.view_layer.objects.active = bpy.data.objects[0]
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
for obj in bpy.data.objects:     # mirroring flipped the winding
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.flip_normals()
    bpy.ops.object.mode_set(mode='OBJECT')

# The distro build registers a stub export_scene.fbx operator without options: call the add-on directly.
from bpy_extras.io_utils import axis_conversion
from io_scene_fbx import export_fbx_bin


class Op:
    def report(self, kind, msg):
        print(kind, msg)


export_fbx_bin.save(
    Op(), bpy.context, filepath=out, global_scale=1.0, apply_scale_options='FBX_SCALE_NONE',
    axis_up='Z', axis_forward='-Y', global_matrix=axis_conversion(to_forward='-Y', to_up='Z').to_4x4(),
    object_types={'MESH'}, mesh_smooth_type='FACE', bake_space_transform=False)
print('wrote', out, [(o.name, len(o.data.polygons)) for o in bpy.data.objects])
