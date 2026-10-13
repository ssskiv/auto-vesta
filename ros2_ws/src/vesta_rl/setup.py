from setuptools import find_packages, setup

package_name = 'vesta_rl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='RL training (2D sim + Webots) for the Vesta neural controller',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'make_paths = vesta_rl.paths:main',
            'train = vesta_rl.train:main',
            'export_onnx = vesta_rl.export_onnx:main',
            'calibrate = vesta_rl.calibrate:main',
            'make_obs_fixture = vesta_rl.make_obs_fixture:main',
            'viewer = vesta_rl.viewer:main',
            'drive = vesta_rl.drive:main',
            'goal_follower = vesta_rl.goal_follower:main',
            'scene_pool = vesta_rl.scene_pool:main',
            'evaluate = vesta_rl.evaluate:main',
        ],
    },
)
