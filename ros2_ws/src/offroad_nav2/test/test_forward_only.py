"""Static guarantees that the Nav2 configuration (config/*.yaml) can never command reverse motion, and that parameter files agree."""
import os
import re

import pytest
import yaml

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BT = os.path.join(PKG, 'behavior_trees')


def build(bt_dir=None, map_yaml=None):
    return yaml.safe_load(open(os.path.join(PKG, 'config', 'nav2_params.yaml')))


def nodes():
    return yaml.safe_load(open(os.path.join(PKG, 'config', 'nodes.yaml')))


def test_params_forward_only():
    p = build(BT, '')
    assert p['planner_server']['ros__parameters']['GridBased']['motion_model_for_search'] == 'DUBIN'
    fp = p['controller_server']['ros__parameters']['FollowPath']
    assert 'RegulatedPurePursuit' in fp['plugin']
    assert fp['allow_reversing'] is False and fp['use_rotate_to_heading'] is False
    vs = p['velocity_smoother']['ros__parameters']
    assert vs['min_velocity'][0] == 0.0 and vs['max_velocity'][0] <= 5.0     # 18 km/h = 5 m/s
    assert p['behavior_server']['ros__parameters']['behavior_plugins'] == ['wait']
    libs = p['bt_navigator']['ros__parameters']['plugin_lib_names']
    assert not any(k in l for l in libs for k in ('back_up', 'spin', 'drive_on_heading', 'assisted_teleop'))


def test_no_lidar_anywhere():
    assert '/lidar' not in str(build(BT, ''))


@pytest.mark.parametrize('f', ['nav_forward_only.xml', 'nav_through_poses_forward_only.xml'])
def test_bt_has_no_reverse_actions(f):
    xml = open(os.path.join(BT, f)).read()
    xml = re.sub(r'<!--.*?-->', '', xml, flags=re.S)
    for bad in ('BackUp', 'Spin', 'DriveOnHeading', 'AssistedTeleop'):
        assert bad not in xml
    assert 'ClearEntireCostmap' in xml and 'Wait' in xml


def test_camera_layers_all_have_clearing_source():
    p = build(BT, '')
    for kind in ('local_costmap', 'global_costmap'):
        c = p[kind][kind]['ros__parameters']
        for name in c['plugins']:
            if name.endswith('_layer') and c[name].get('plugin', '').endswith('ObstacleLayer'):
                srcs = c[name]['observation_sources'].split()
                assert any(c[name][s]['clearing'] for s in srcs), name
    assert 'semantic_layer' not in p['global_costmap']['global_costmap']['ros__parameters']['plugins']


def test_nodes_yaml_consistent():
    n = nodes()
    # every node uses simulator time; the stereo crop of the relay and of the depth converter must agree
    for name, v in n.items():
        if name != 'cmd_to_ackermann':   # real-car path only
            assert v['ros__parameters']['use_sim_time'] is True, name
    assert n['stamp_relay']['ros__parameters']['crop_top'] == n['disparity_to_depth']['ros__parameters']['crop_top']
    # the relay in the command chain: input is the collision monitor output, output is the simulator command
    z = n['zero_clamp_relay']['ros__parameters']
    nav = build()
    assert z['in_topic'].lstrip('/') == nav['collision_monitor']['ros__parameters']['cmd_vel_out_topic']
    # the navsat datum of the sim adapter and of navsat_transform must be the same point
    d = yaml.safe_load(open(os.path.join(PKG, 'config', 'navsat.yaml')))['navsat_transform']['ros__parameters']['datum']
    a = n['sim_sensor_adapter']['ros__parameters']
    assert abs(d[0] - a['datum_lat']) < 1e-9 and abs(d[1] - a['datum_lon']) < 1e-9


def test_substituted_paths_are_names_only():
    nav = build()['bt_navigator']['ros__parameters']
    assert nav['default_nav_to_pose_bt_xml'] == 'nav_forward_only.xml'
    assert os.path.exists(os.path.join(BT, nav['default_nav_to_pose_bt_xml']))
    assert os.path.exists(os.path.join(BT, nav['default_nav_through_poses_bt_xml']))
