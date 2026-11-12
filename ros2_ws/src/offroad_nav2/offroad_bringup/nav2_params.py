"""Nav2 (Humble 1.1.x) parameters, forward-only. Built in Python so the obstacle source (camera models vs lidar) and
the vehicle numbers (offroad_description/config/vehicle.yaml) switch cleanly.  `python3 -m offroad_bringup.nav2_params
camera > nav2_params.camera.yaml` dumps the file for inspection.

Forward-only everywhere: Smac Hybrid DUBIN, MPPI vx_min 0, smoother min_velocity x 0, BT without BackUp/Spin,
collision_monitor + zero-clamp relay (offroad_vehicle_interface). Heights are in the costmap frame (odom/map): the
ground sits at z = -base_link_height (EKF 2D mode keeps base_link at z = 0).
"""
import copy

import yaml

FOOTPRINT = '[[3.68, 0.865], [3.68, -0.865], [-0.88, -0.865], [-0.88, 0.865]]'   # vehicle.yaml, rear axle origin
R_MIN = 4.0            # TODO(user) decision 7.7: guide suggests 4.5 vs 5.2; 4.0 tried because forward-only corner traps (kinematic min 3.87 m at 0.6 rad)
V_MAX = 4.5            # RoboCross limit 5.0 m/s = 18 km/h
GROUND_Z = -0.108      # = -base_link_height (measured in Webots)

CLEAR = ('clear', '/perception/points/clear', False, True)   # drivable-ground points: rays clear stale marks
# Every obstacle layer keeps its OWN cell buffer, so each one needs the clearing source to forget stale marks.
CAMERA_SOURCES = {     # layer -> list of (source name, topic, marking, clearing)
    'geometric_layer': [('geometric', '/perception/points/geometric', True, False), CLEAR],
    'vru_layer': [('vru', '/perception/points/vru', True, False), CLEAR],
    'vehicle_layer': [('vehicle', '/perception/points/vehicle', True, False), CLEAR],
    # semantic layer (local costmap only): flat no-go ground + natural/man-made obstacles. TODO(user) decision 7.10:
    # lethal vs soft. Kept out of the global costmap: a misclassified patch would block the corridor for good.
    'semantic_layer': [('nondrivable', '/perception/points/nondrivable', True, False),
                       ('natural', '/perception/points/natural', True, False),
                       ('manmade', '/perception/points/manmade', True, False), CLEAR],
}


def obs_layer(sources, lidar=False):
    d = {'plugin': 'nav2_costmap_2d::ObstacleLayer', 'enabled': True,
         'observation_sources': ' '.join(s[0] for s in sources)}
    for name, topic, marking, clearing in sources:
        d[name] = {
            'topic': topic, 'data_type': 'PointCloud2', 'marking': marking, 'clearing': clearing,
            'sensor_frame': '', 'observation_persistence': 0.0, 'expected_update_rate': 0.0,
            'inf_is_valid': False, 'obstacle_max_range': 20.0 if not lidar else 30.0, 'obstacle_min_range': 0.0,
            'raytrace_max_range': 22.0 if not lidar else 32.0, 'raytrace_min_range': 0.0,
            # heights are in the costmap frame; clouds are already height-filtered against the fitted ground plane
            'min_obstacle_height': GROUND_Z + (0.15 if lidar else -0.4), 'max_obstacle_height': 3.0,
        }
    return d


def inflation(radius, scale=3.0):
    return {'plugin': 'nav2_costmap_2d::InflationLayer', 'cost_scaling_factor': scale, 'inflation_radius': radius}


def costmap(kind, mode):
    c = {'update_frequency': 10.0 if kind == 'local' else 2.0, 'publish_frequency': 5.0 if kind == 'local' else 1.0,
         'global_frame': 'odom' if kind == 'local' else 'map', 'robot_base_frame': 'base_link',
         'footprint': FOOTPRINT, 'footprint_padding': 0.2,   # planning margin (0.12: 1-3 barrel touches/lap; 0.3: planner pinned the car against a barrel)
          'resolution': 0.1 if kind == 'local' else 0.2,
         'transform_tolerance': 0.3, 'always_send_full_costmap': True, 'use_sim_time': True}
    if kind == 'local':
        c.update({'rolling_window': True, 'width': 40, 'height': 40})
    else:
        c.update({'rolling_window': False, 'track_unknown_space': True})
    layers = {}
    if mode == 'camera':
        order = (['vru_layer', 'vru_inflation', 'vehicle_layer', 'vehicle_inflation', 'geometric_layer', 'semantic_layer',
                  'inflation_layer'] if kind == 'local' else
                 ['static_layer', 'vru_layer', 'vehicle_layer', 'geometric_layer', 'inflation_layer'])
        for n, srcs in CAMERA_SOURCES.items():
            layers[n] = obs_layer(srcs)
        layers['vru_inflation'] = inflation(3.0, 1.5)        # VRU halo larger than the vehicle halo (check in rviz)
        layers['vehicle_inflation'] = inflation(2.0, 2.0)
    else:   # lidar-only ablation
        order = ['lidar_layer', 'inflation_layer'] if kind == 'local' else ['static_layer', 'lidar_layer', 'inflation_layer']
        layers['lidar_layer'] = obs_layer([('lidar', '/lidar', True, True)], lidar=True)
    layers['static_layer'] = {'plugin': 'nav2_costmap_2d::StaticLayer', 'map_subscribe_transient_local': True}
    # final inflation. MPPI CostCritic wants radius >= half the largest footprint dimension (4.56 m / 2 = 2.28 m) for
    # corner-only footprint checks; a steep decay keeps the slalom gaps open (cost at 1.6 m from the hull: ~14 of 253).
    layers['inflation_layer'] = inflation(2.4, 4.0)
    c['plugins'] = order
    c.update({k: v for k, v in layers.items() if k in order})
    return c


def build(mode='camera', bt_dir='/tmp', map_yaml='', controller='mppi'):
    assert mode in ('camera', 'lidar') and controller in ('mppi', 'rpp')
    p = {}
    p['planner_server'] = {'ros__parameters': {
        'use_sim_time': True, 'expected_planner_frequency': 2.0, 'planner_plugins': ['GridBased'],
        'GridBased': {
            'plugin': 'nav2_smac_planner/SmacPlannerHybrid', 'downsample_costmap': True, 'downsampling_factor': 2,   # 0.4 m search grid: the 4.6 m footprint search was too slow
            'tolerance': 3.0, 'allow_unknown': True, 'max_iterations': 2000000, 'max_on_approach_iterations': 1000,
            'max_planning_time': 5.0, 'motion_model_for_search': 'DUBIN', 'angle_quantization_bins': 48,
            'analytic_expansion_ratio': 3.5, 'analytic_expansion_max_length': 3.0, 'minimum_turning_radius': R_MIN,
            'reverse_penalty': 2.0, 'change_penalty': 0.0, 'non_straight_penalty': 1.2, 'cost_penalty': 2.0,
            'retrospective_penalty': 0.015, 'lookup_table_size': 20.0, 'cache_obstacle_heuristic': False,
            'smooth_path': True,
            'smoother': {'max_iterations': 1000, 'w_smooth': 0.3, 'w_data': 0.2, 'tolerance': 1e-10,
                         'do_refinement': True, 'refinement_num': 2}}}}
    p['controller_server'] = {'ros__parameters': {
        'use_sim_time': True, 'controller_frequency': 10.0, 'min_x_velocity_threshold': 0.001,
        'min_y_velocity_threshold': 0.5, 'min_theta_velocity_threshold': 0.001, 'failure_tolerance': 0.3,
        'progress_checker_plugin': 'progress_checker', 'goal_checker_plugins': ['general_goal_checker'],
        'controller_plugins': ['FollowPath'],
        'progress_checker': {'plugin': 'nav2_controller::SimpleProgressChecker', 'required_movement_radius': 0.5,
                             'movement_time_allowance': 45.0},
        'general_goal_checker': {'stateful': True, 'plugin': 'nav2_controller::SimpleGoalChecker',
                                 'xy_goal_tolerance': 1.5, 'yaw_goal_tolerance': 3.14},
        'FollowPath': {
            'plugin': 'nav2_mppi_controller::MPPIController', 'time_steps': 30, 'model_dt': 0.1, 'batch_size': 800,   # 3 s horizon; heavier settings missed 20 Hz here
            
            'vx_std': 0.3, 'vy_std': 0.2, 'wz_std': 0.4, 'vx_max': V_MAX, 'vx_min': 0.0, 'vy_max': 0.0, 'wz_max': 1.0,
            'iteration_count': 1, 'prune_distance': 1.7, 'transform_tolerance': 0.1, 'temperature': 0.3,
            'gamma': 0.015, 'motion_model': 'Ackermann', 'visualize': True, 'reset_period': 1.0,
            'regenerate_noises': True,
            'TrajectoryVisualizer': {'trajectory_step': 5, 'time_step': 3},
            'AckermannConstraints': {'min_turning_r': R_MIN},
            'critics': ['ConstraintCritic', 'CostCritic', 'GoalCritic', 'PathAlignCritic', 'PathFollowCritic',
                        'PathAngleCritic', 'PreferForwardCritic'],
            'ConstraintCritic': {'enabled': True, 'cost_power': 1, 'cost_weight': 4.0},
            'GoalCritic': {'enabled': True, 'cost_power': 1, 'cost_weight': 5.0, 'threshold_to_consider': 1.4},
            'CostCritic': {'enabled': True, 'cost_power': 1, 'cost_weight': 3.81, 'critical_cost': 300.0,
                           'consider_footprint': True, 'collision_cost': 1000000.0, 'near_goal_distance': 1.0,
                           'trajectory_point_step': 2},
            'PathAlignCritic': {'enabled': True, 'cost_power': 1, 'cost_weight': 14.0, 'max_path_occupancy_ratio': 0.05,
                                'trajectory_point_step': 4, 'threshold_to_consider': 0.5, 'offset_from_furthest': 20,
                                'use_path_orientations': False},
            'PathFollowCritic': {'enabled': True, 'cost_power': 1, 'cost_weight': 5.0, 'offset_from_furthest': 5,
                                 'threshold_to_consider': 1.4},
            'PathAngleCritic': {'enabled': True, 'cost_power': 1, 'cost_weight': 2.0, 'offset_from_furthest': 4,
                                'threshold_to_consider': 0.5, 'max_angle_to_furthest': 1.0},
            'PreferForwardCritic': {'enabled': True, 'cost_power': 1, 'cost_weight': 5.0,
                                    'threshold_to_consider': 0.5}}}}
    if controller == 'rpp':   # fallback / comparison: the planner path is already curvature-feasible (Dubin)
        p['controller_server']['ros__parameters']['FollowPath'] = {
            'plugin': 'nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController',
            'desired_linear_vel': 3.0, 'lookahead_dist': 3.0, 'min_lookahead_dist': 2.0, 'max_lookahead_dist': 5.0,
            'lookahead_time': 1.0,   # shorter lookahead: less corner cutting past barrels
            'rotate_to_heading_angular_vel': 0.0, 'transform_tolerance': 0.1,
            'use_velocity_scaled_lookahead_dist': True, 'min_approach_linear_velocity': 1.0,
            'approach_velocity_scaling_dist': 4.0, 'use_collision_detection': True,
            'max_allowed_time_to_collision_up_to_carrot': 1.0, 'use_regulated_linear_velocity_scaling': True,
            'use_cost_regulated_linear_velocity_scaling': True, 'regulated_linear_scaling_min_radius': 6.0,
            'regulated_linear_scaling_min_speed': 1.5, 'use_rotate_to_heading': False, 'allow_reversing': False,
            'max_angular_accel': 3.2, 'cost_scaling_dist': 1.0, 'cost_scaling_gain': 1.0, 'inflation_cost_scaling_factor': 4.0}
    p['local_costmap'] = {'local_costmap': {'ros__parameters': costmap('local', mode)}}
    p['global_costmap'] = {'global_costmap': {'ros__parameters': costmap('global', mode)}}
    p['map_server'] = {'ros__parameters': {'use_sim_time': True, 'yaml_filename': map_yaml}}
    p['behavior_server'] = {'ros__parameters': {
        'use_sim_time': True, 'costmap_topic': 'local_costmap/costmap_raw',
        'footprint_topic': 'local_costmap/published_footprint', 'cycle_frequency': 10.0,
        'behavior_plugins': ['wait'], 'wait': {'plugin': 'nav2_behaviors/Wait'},   # no BackUp, no Spin
        'global_frame': 'odom', 'robot_base_frame': 'base_link', 'transform_tolerance': 0.1,
        'simulate_ahead_time': 2.0}}
    libs = ['nav2_compute_path_to_pose_action_bt_node', 'nav2_compute_path_through_poses_action_bt_node',
            'nav2_follow_path_action_bt_node', 'nav2_wait_action_bt_node', 'nav2_clear_costmap_service_bt_node',
            'nav2_is_stuck_condition_bt_node', 'nav2_goal_reached_condition_bt_node',
            'nav2_goal_updated_condition_bt_node', 'nav2_globally_updated_goal_condition_bt_node',
            'nav2_is_path_valid_condition_bt_node', 'nav2_initial_pose_received_condition_bt_node',
            'nav2_rate_controller_bt_node', 'nav2_distance_controller_bt_node', 'nav2_speed_controller_bt_node',
            'nav2_truncate_path_action_bt_node', 'nav2_truncate_path_local_action_bt_node',
            'nav2_goal_updater_node_bt_node', 'nav2_recovery_node_bt_node', 'nav2_pipeline_sequence_bt_node',
            'nav2_round_robin_node_bt_node', 'nav2_transform_available_condition_bt_node',
            'nav2_time_expired_condition_bt_node', 'nav2_distance_traveled_condition_bt_node']
    p['bt_navigator'] = {'ros__parameters': {
        'use_sim_time': True, 'global_frame': 'map', 'robot_base_frame': 'base_link', 'odom_topic': '/odometry/local',
        'bt_loop_duration': 10, 'default_server_timeout': 20, 'plugin_lib_names': libs,
        'default_nav_to_pose_bt_xml': f'{bt_dir}/nav_forward_only.xml',
        'default_nav_through_poses_bt_xml': f'{bt_dir}/nav_through_poses_forward_only.xml'}}
    p['waypoint_follower'] = {'ros__parameters': {
        'use_sim_time': True, 'loop_rate': 20, 'stop_on_failure': False,
        'waypoint_task_executor_plugin': 'wait_at_waypoint',
        'wait_at_waypoint': {'plugin': 'nav2_waypoint_follower::WaitAtWaypoint', 'enabled': True,
                             'waypoint_pause_duration': 0}}}
    p['velocity_smoother'] = {'ros__parameters': {
        'use_sim_time': True, 'smoothing_frequency': 20.0, 'scale_velocities': False, 'feedback': 'OPEN_LOOP',
        'max_velocity': [V_MAX, 0.0, 1.0], 'min_velocity': [0.0, 0.0, -1.0],      # x never negative
        'max_accel': [1.5, 0.0, 2.0], 'max_decel': [-3.0, 0.0, -2.0], 'odom_topic': '/odometry/local',
        'odom_duration': 0.1, 'deadband_velocity': [0.0, 0.0, 0.0], 'velocity_timeout': 1.0}}
    # last resort only: a tight stop zone at the bumper (hull x 3.68, camera blind below ~x 3.95) and a wider slow-down
    # zone. A big stop zone dead ahead would freeze a forward-only car that cannot steer in place.
    stop = [4.7, 0.85, 4.7, -0.85, 3.75, -0.85, 3.75, 0.85]   # = vehicle half-width: only what we would really hit
    slow = [7.5, 1.1, 7.5, -1.1, 3.75, -1.1, 3.75, 1.1]
    p['collision_monitor'] = {'ros__parameters': {
        'use_sim_time': True, 'base_frame_id': 'base_link', 'odom_frame_id': 'odom',
        'cmd_vel_in_topic': 'cmd_vel_smoothed', 'cmd_vel_out_topic': 'cmd_vel_nav', 'transform_tolerance': 0.3,
        'source_timeout': 2.0, 'base_shift_correction': True, 'stop_pub_timeout': 2.0,
        'polygons': ['StopZone', 'SlowZone'],
        'StopZone': {'type': 'polygon', 'points': stop, 'action_type': 'stop', 'min_points': 4,
                     'visualize': True, 'polygon_pub_topic': 'stop_polygon', 'enabled': True},
        'SlowZone': {'type': 'polygon', 'points': slow, 'action_type': 'slowdown', 'min_points': 4,
                     'slowdown_ratio': 0.5, 'visualize': True, 'polygon_pub_topic': 'slow_polygon', 'enabled': True},
        'observation_sources': ['stop_wall'] + (['perception_geo'] if mode == 'camera' else ['lidar_pc']),
        'stop_wall': {'type': 'pointcloud', 'topic': '/perception/stop_wall', 'min_height': -1.0, 'max_height': 3.0},
        'perception_geo': {'type': 'pointcloud', 'topic': '/perception/points/geometric', 'min_height': -0.05,
                           'max_height': 2.0},
        'lidar_pc': {'type': 'pointcloud', 'topic': '/lidar', 'min_height': 0.1, 'max_height': 2.0}}}
    p['lifecycle_manager_navigation'] = {'ros__parameters': {
        'use_sim_time': True, 'autostart': True, 'bond_timeout': 8.0,
        'node_names': ['map_server', 'planner_server', 'controller_server', 'behavior_server', 'bt_navigator',
                       'waypoint_follower', 'velocity_smoother', 'collision_monitor']}}
    return copy.deepcopy(p)


if __name__ == '__main__':
    import sys
    print(yaml.safe_dump(build(sys.argv[1] if len(sys.argv) > 1 else 'camera'), sort_keys=False))
