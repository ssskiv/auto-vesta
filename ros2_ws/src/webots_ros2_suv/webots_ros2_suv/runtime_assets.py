"""Connect mirrored PROTOs to the active Webots controllers and plugins."""

import os
from pathlib import Path


def link_runtime_assets(package_dir: Path) -> None:
    """Create or refresh installation-specific links without replacing copies."""
    webots_home = Path(os.environ.get('WEBOTS_HOME', '/usr/local/webots'))
    for relative in (
        'projects/objects/traffic/controllers',
        'projects/vehicles/plugins',
    ):
        link = package_dir / 'protos/webots' / relative
        target = webots_home / relative
        if link.is_symlink():
            if link.readlink() == target:
                continue
            link.unlink()
        elif link.exists():
            # Older colcon installs contain actual copies of these directories.
            continue
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
