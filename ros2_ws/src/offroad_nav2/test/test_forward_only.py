"""Static guarantees that the generated Nav2 configuration can never command reverse motion."""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from offroad_bringup.nav2_params import build  # noqa: E402

BT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'behavior_trees')


@pytest.mark.parametrize('controller', ['mppi', 'rpp'])
@pytest.mark.parametrize('mode', ['camera', 'lidar'])
def test_params_forward_only(mode, controller):
    p = build(mode, BT, '', controller)
    assert p['planner_server']['ros__parameters']['GridBased']['motion_model_for_search'] == 'DUBIN'
    fp = p['controller_server']['ros__parameters']['FollowPath']
    if controller == 'mppi':
        assert fp['vx_min'] == 0.0 and fp['motion_model'] == 'Ackermann' and 'PreferForwardCritic' in fp['critics']
    else:
        assert fp['allow_reversing'] is False and fp['use_rotate_to_heading'] is False
    vs = p['velocity_smoother']['ros__parameters']
    assert vs['min_velocity'][0] == 0.0 and vs['max_velocity'][0] <= 5.0     # 18 km/h = 5 m/s
    assert p['behavior_server']['ros__parameters']['behavior_plugins'] == ['wait']
    libs = p['bt_navigator']['ros__parameters']['plugin_lib_names']
    assert not any(k in l for l in libs for k in ('back_up', 'spin', 'drive_on_heading', 'assisted_teleop'))


@pytest.mark.parametrize('f', ['nav_forward_only.xml', 'nav_through_poses_forward_only.xml'])
def test_bt_has_no_reverse_actions(f):
    xml = open(os.path.join(BT, f)).read()
    xml = re.sub(r'<!--.*?-->', '', xml, flags=re.S)
    for bad in ('BackUp', 'Spin', 'DriveOnHeading', 'AssistedTeleop'):
        assert bad not in xml
    assert 'ClearEntireCostmap' in xml and 'Wait' in xml


def test_camera_layers_all_have_clearing_source():
    p = build('camera', BT, '')
    for kind in ('local_costmap', 'global_costmap'):
        c = p[kind][kind]['ros__parameters']
        for name in c['plugins']:
            if name.endswith('_layer') and c[name].get('plugin', '').endswith('ObstacleLayer'):
                srcs = c[name]['observation_sources'].split()
                assert any(c[name][s]['clearing'] for s in srcs), name
    assert 'semantic_layer' not in p['global_costmap']['global_costmap']['ros__parameters']['plugins']
