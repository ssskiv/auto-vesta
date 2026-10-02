import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'webots_ros2_suv'


def package_files(src_dir):
    """(share/<package>/<rel_dir>, [files]) tuples for every file under src_dir."""
    files_by_dest = {}
    for path in glob(os.path.join(src_dir, '**', '*'), recursive=True):
        if os.path.isfile(path):
            dest = os.path.join('share', package_name, os.path.relpath(os.path.dirname(path), '.'))
            files_by_dest.setdefault(dest, []).append(path)
    return list(files_by_dest.items())


data_files = [
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]
for directory in ('launch', 'worlds', 'protos', 'resource'):
    data_files += package_files(directory)

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
