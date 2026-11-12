from setuptools import find_packages, setup

package_name = 'carla_suv'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/carla.launch.py']),
        ('share/' + package_name + '/config', ['config/vesta_sensors.json', 'config/vesta_stereo_sensors.json']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ivan Ioshpa',
    maintainer_email='ivanioshpa@gmail.com',
    description='CARLA simulation of the Lada Vesta',
    license='Apache License 2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'cmd_vel_adapter = carla_suv.cmd_vel_adapter:main',
            'sensor_adapter = carla_suv.sensor_adapter:main',
            'world_setup = carla_suv.world_setup:main',
        ],
    },
)
