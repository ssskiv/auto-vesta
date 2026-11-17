"""Check the shared driver supports vehicles with and without radar."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from webots_ros2_suv import vesta_driver


@pytest.mark.parametrize('has_radar', [False, True])
def test_optional_radar_init_and_step(monkeypatch, has_radar):
    robot = MagicMock()
    robot.getBasicTimeStep.return_value = 8
    radar = MagicMock() if has_radar else None
    robot.getDevice.side_effect = lambda name: radar if name == 'radar' else MagicMock()
    node = MagicMock()
    monkeypatch.setattr(vesta_driver.rclpy, 'init', MagicMock())
    monkeypatch.setattr(vesta_driver.rclpy, 'create_node', lambda *a, **kw: node)
    monkeypatch.setattr(vesta_driver.rclpy, 'spin_once', MagicMock())
    monkeypatch.setattr(vesta_driver, 'TransformBroadcaster', MagicMock())
    monkeypatch.setenv('VESTA_STANDIN_TL', '0')

    driver = vesta_driver.VestaDriver()
    driver.init(SimpleNamespace(robot=robot), {'wheelbase': '1.799622521265'})
    topics = [call.args[1] for call in node.create_publisher.call_args_list]
    assert ('/radar' in topics) == has_radar
    assert '/odom' in topics
    assert '/imu/data' in topics
    if has_radar:
        radar.enable.assert_called_once_with(8)

    cloud = MagicMock()
    monkeypatch.setattr(driver, '_VestaDriver__radar_point_cloud', cloud)
    for method in ('joint_states', 'base_link_transform', 'odometry'):
        monkeypatch.setattr(driver, f'_VestaDriver__{method}', MagicMock())
    driver.step()
    assert cloud.call_count == int(has_radar)
    if has_radar:
        node.create_publisher.return_value.publish.assert_any_call(cloud.return_value)
