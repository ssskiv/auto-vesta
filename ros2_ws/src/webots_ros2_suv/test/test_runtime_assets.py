from pathlib import Path

import pytest

from webots_ros2_suv.runtime_assets import link_runtime_assets


@pytest.mark.parametrize('relative', [
    'projects/objects/traffic/controllers',
    'projects/vehicles/plugins',
])
@pytest.mark.parametrize('stale', [False, True])
def test_runtime_assets_follow_webots_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str, stale: bool,
) -> None:
    package = tmp_path / 'relocated checkout'
    installation = tmp_path / 'custom webots'
    target = installation / relative
    target.mkdir(parents=True)
    (target / 'asset').write_text('installed controller or plugin')
    link = package / 'protos/webots' / relative
    if stale:
        link.parent.mkdir(parents=True)
        link.symlink_to(tmp_path / 'old installation' / relative)
    monkeypatch.setenv('WEBOTS_HOME', str(installation))

    link_runtime_assets(package)

    assert (link / 'asset').read_text() == 'installed controller or plugin'


def test_runtime_assets_preserve_installed_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    controllers = tmp_path / 'protos/webots/projects/objects/traffic/controllers'
    controllers.mkdir(parents=True)
    (controllers / 'asset').write_text('packaged controller')
    monkeypatch.setenv('WEBOTS_HOME', str(tmp_path / 'custom webots'))

    link_runtime_assets(tmp_path)

    assert (controllers / 'asset').read_text() == 'packaged controller'
