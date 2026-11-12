"""perception_node (C++, three networks via ONNX Runtime + depth -> clouds), the traffic-light node and the stop wall."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('tl_source', default_value='none'),   # none | standin
        Node(package='offroad_nav2', executable='perception_node', name='perception_node', output='screen',
             parameters=[{'use_sim_time': True,
                          # ONNX graphs exported from the checkpoints by tools/export_onnx.py (seg/contour/objseg.onnx)
                          'onnx_dir': '/models/onnx',
                          'preprocess_mode': 'resize',      # TODO(user) decision 7.8 (resize vs crop)
                          }]),
        Node(package='offroad_nav2', executable='traffic_light_node', name='traffic_light_node', output='screen',
             parameters=[{'use_sim_time': True, 'source': LaunchConfiguration('tl_source')}]),
        Node(package='offroad_nav2', executable='stop_wall_node', name='stop_wall', output='screen',
             parameters=[{'use_sim_time': True, 'enabled': PythonExpression(["'", LaunchConfiguration('tl_source'), "' != 'none'"]),
                          # RoboCross stop line: world (11.40, -0.14) = map (9.20, 1.46), across the track (map = world - (2.2, -1.6))
                          'line_xy': [9.20, 1.46], 'line_yaw': 0.0, 'width': 12.0, 'wall_offset': 4.0}]),
    ])
