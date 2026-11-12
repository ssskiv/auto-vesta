"""Unreal Python: copy stock barrier / curb static meshes into /Game/robocross/Props to place by hand.

    UE4Editor CarlaUE4.uproject -run=pythonscript -script=/path/ue_copy_props.py
Idempotent: existing copies are skipped. Editor must be closed.
"""
import unreal

DEST = '/Game/robocross/Props'
S = '/Game/Carla/Static/'
SOURCES = [
    'Static/SM_Barrier', 'Static/SM_ChainBarrier', 'Static/SM_ChainBarrierEnd',
    'Static/SM_ParkingBarrier', 'Static/SM_ParkingBarrier_Short',
    'Dynamic/Construction/SM_StreetBarrier', 'Dynamic/Construction/SM_TrafficCones_01',
    'Dynamic/Construction/SM_TrafficCones_02', 'Dynamic/Construction/SM_TrafficCones_03',
    'Dynamic/Construction/SM_ConstructionCone',
    'GuardRail/SM_GuardRail_high', 'GuardRail/SM_Guardrail_02',
    'SideWalk/SideWalkCube', 'SideWalk/SidewalkPlane', 'SideWalk/St_PathWay_Block', 'SideWalk/St_PathWay_Block1',
    'SideWalk/St_Road_TileRoad_SidewalkL', 'SideWalk/St_Road_TileRoad_SidewalkR',
]
assets = unreal.EditorAssetLibrary
assets.make_directory(DEST)
for rel in SOURCES:
    name = rel.split('/')[-1]
    src, dst = S + rel, '%s/%s' % (DEST, name)
    if assets.does_asset_exist(dst):
        unreal.log_warning('props: exists ' + name)
    elif not assets.does_asset_exist(src):
        unreal.log_warning('props: MISSING ' + src)
    elif assets.duplicate_asset(src, dst) is None:
        unreal.log_warning('props: FAILED ' + name)
    else:
        unreal.log_warning('props: copied ' + name)
assets.save_directory(DEST)
unreal.log_warning('props: done')
