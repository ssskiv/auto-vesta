import math
import os
import time

import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.parameter import Parameter
from sensor_msgs.msg import Imu, JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64, Header, Int32
from tf2_ros import TransformBroadcaster
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy


import traceback

ODOM_FRAME = 'odom'
BASE_FRAME = 'base_link'
RADAR_FRAME = 'radar_link'
LIDAR_FRAME = 'lidar_link'

WHEEL_SENSOR_JOINTS = (
    ('left_front_sensor', 'front_left_wheel_joint'),
    ('right_front_sensor', 'front_right_wheel_joint'),
    ('left_rear_sensor', 'rear_left_wheel_joint'),
    ('right_rear_sensor', 'rear_right_wheel_joint'),
)
STEER_JOINTS = ('front_left_steer_joint', 'front_right_steer_joint')

COMMAND_TIMEOUT = 0.5
WHEELBASE = 2.648
MAX_STEER = 0.6

def _clamp(value, low, high):
    return max(low, min(high, value))


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class VestaDriver:
    def init(self, webots_node, properties):
        self.__driver = webots_node.robot
        self.__driver.setGear(1)
        self.__driver.setThrottle(0.0)
        self.__driver.setBrakeIntensity(0.0)
        self.__driver.setSteeringAngle(0.0)

        
        
        # 'torque' mode: raw cmd_throttle/cmd_brake -> setThrottle/setBrakeIntensity.
        # 'speed' mode: cmd_vel -> setCruisingSpeed (closed-loop, so a small
        # linear.x like teleop_twist_keyboard's 0.5 m/s still moves the car,
        # unlike an open-loop throttle fraction would).
        self.__mode = 'torque'
        self.__throttle = 0.0
        self.__brake = 0.0
        self.__steer = 0.0
        self.__cruising_speed_kmh = 0.0
        self.__last_cmd_time = time.monotonic()
        self.__orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self.__yaw_rate = 0.0
        self.__gps_origin = None
        self.__teleport_request = None
        self.__barrels_request = None
        self.__barrels_countdown = 0


        qos = qos_profile_sensor_data
        qos.reliability = QoSReliabilityPolicy.RELIABLE

        timestep = int(self.__driver.getBasicTimeStep())
        self.__radar = self.__driver.getDevice('radar')
        self.__radar.enable(timestep)
        self.__gps = self.__driver.getDevice('gps')
        self.__gps.enable(timestep)
        self.__lidar = self.__driver.getDevice('lidar')
        self.__lidar.enable(timestep)
        self.__lidar.enablePointCloud()

        for imu_device_name in ('imu', 'imu_gyro', 'imu_accelerometer'):
            self.__driver.getDevice(imu_device_name).enable(timestep)

        self.__wheel_sensors = []
        for device_name, joint_name in WHEEL_SENSOR_JOINTS:
            sensor = self.__driver.getDevice(device_name)
            sensor.enable(timestep)
            self.__wheel_sensors.append((sensor, joint_name))

        rclpy.init(args=None)
        self.__node = rclpy.create_node(
            'vesta_driver', parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.__node.create_subscription(Twist, '/cmd_vel', self.__on_cmd_vel, qos)
        self.__node.create_subscription(Imu, '/vesta/imu/data', self.__on_imu, qos)
        self.__node.create_subscription(PointCloud2, '/vesta/lidar/point_cloud', self.__on_point_cloud, qos)
        # RL resets: move the car to a pose given in the odom frame (the robot
        # node has supervisor TRUE and Driver is a Supervisor subclass).
        self.__node.create_subscription(PoseStamped, '/vesta/teleport', self.__on_teleport, 1)
        # New barrel layout (RobocrossBarrels.proto) with this randomSeed (<= 0: random).
        self.__node.create_subscription(Int32, '/vesta/barrels_seed', self.__on_barrels_seed, 1)
        
        # VESTA_PUBLISH_ODOM_TF=0 (set by a localization stack that owns odom->base_link): do not broadcast it.
        self.__publish_odom_tf = os.environ.get('VESTA_PUBLISH_ODOM_TF', '1').lower() not in ('0', 'false', 'no')
        self.__tf_broadcaster = TransformBroadcaster(self.__node)
        self.__imu_publisher = self.__node.create_publisher(Imu, '/imu/data', qos)
        self.__lidar_publisher = self.__node.create_publisher(PointCloud2, '/lidar', qos)
        self.__radar_publisher = self.__node.create_publisher(PointCloud2, '/radar', qos)
        self.__joint_state_pub = self.__node.create_publisher(JointState, '/joint_states', qos)
        self.__odom_pub = self.__node.create_publisher(Odometry, '/odom', qos)


    def __sim_stamp(self):
        """Webots simulation time (== /clock), exact for this step, independent of how far the node's own clock lags."""
        t = self.__driver.getTime()
        sec = int(t)
        return Time(sec=sec, nanosec=int((t - sec) * 1e9))

    def __on_imu(self, msg):
        try:
            p = msg
            p.header.stamp = self.__sim_stamp()
            self.__imu_publisher.publish(p)
            self.__orientation = msg.orientation
            self.__yaw_rate = msg.angular_velocity.z
        except  Exception as err:
            self.__node._logger.error(''.join(traceback.TracebackException.from_exception(err).format()))
            

    def __on_cmd_vel(self, msg):
        """Drive from a generic Twist (teleop_twist_keyboard, nav2, ...)."""
        self.__mode = 'speed'
        linear_x = msg.linear.x
        self.__cruising_speed_kmh = linear_x * 3.6
        self.__brake = 0.0
        if abs(linear_x) > 1e-3:
            # atan, not atan2: atan2 maps any negative linear_x to ~+-pi (full lock when reversing).
            self.__steer = _clamp(math.atan(-msg.angular.z * WHEELBASE / linear_x), -MAX_STEER, MAX_STEER)
        self.__last_cmd_time = time.monotonic()

    def __on_teleport(self, msg):
        self.__teleport_request = msg.pose
    def __on_barrels_seed(self, msg):
        self.__barrels_request = msg.data

    def step(self):
        # spin_once handles ONE callback; with imu, lidar, /clock and /cmd_vel pending it lagged behind by several steps
        for _ in range(16):
            rclpy.spin_once(self.__node, timeout_sec=0)

        if self.__teleport_request is not None:
            self.__teleport(self.__teleport_request)
            self.__teleport_request = None
        if self.__barrels_request is not None:
            if self.__barrels_countdown == 0:
                self.__barrels_countdown = self.__announce_reseed(self.__barrels_request)
            else:
                self.__barrels_countdown -= 1
            if self.__barrels_countdown == 0:
                self.__reseed_barrels(self.__barrels_request)
                self.__barrels_request = None

        if time.monotonic() - self.__last_cmd_time > COMMAND_TIMEOUT:
            self.__driver.setThrottle(0.0)
            self.__driver.setCruisingSpeed(0.0)
            self.__driver.setBrakeIntensity(1.0)
        elif self.__mode == 'speed':
            self.__driver.setCruisingSpeed(self.__cruising_speed_kmh)
            self.__driver.setBrakeIntensity(self.__brake)
        else:
            # Brake overrides throttle so the two actuators never fight each other.
            throttle = 0.0 if self.__brake > 0.0 else self.__throttle
            self.__driver.setGear(-1 if throttle < 0.0 else 1)
            self.__driver.setThrottle(abs(throttle))
            self.__driver.setBrakeIntensity(self.__brake)

        self.__driver.setSteeringAngle(self.__steer)
        self.__radar_publisher.publish(self.__radar_point_cloud())
        self.__joint_state_pub.publish(self.__joint_states())
        base_link = self.__base_link_transform()
        if self.__publish_odom_tf:
            self.__tf_broadcaster.sendTransform(base_link)
        self.__odom_pub.publish(self.__odometry(base_link))

    def __odometry(self, base_link):
        """/odom with the same pose as the odom->base_link TF.

        Forward speed is the GPS speed vector projected on the heading (signed,
        so reversing gives v < 0); yaw rate comes from the IMU gyro.
        """
        msg = Odometry()
        msg.header = base_link.header
        msg.child_frame_id = BASE_FRAME
        msg.pose.pose.position.x = base_link.transform.translation.x
        msg.pose.pose.position.y = base_link.transform.translation.y
        msg.pose.pose.position.z = base_link.transform.translation.z
        msg.pose.pose.orientation = base_link.transform.rotation

        speed = self.__gps.getSpeedVector()
        if not any(math.isnan(v) for v in speed):
            yaw = _yaw(self.__orientation)
            msg.twist.twist.linear.x = speed[0] * math.cos(yaw) + speed[1] * math.sin(yaw)
        msg.twist.twist.angular.z = self.__yaw_rate
        return msg

    def __teleport(self, pose):
        """Put base_link (= the GPS antenna, see __base_link_transform) at an odom pose."""
        if self.__gps_origin is None:
            return
        root = self.__driver.getSelf()
        root_pos = root.getPosition()
        gps = self.__gps.getValues()
        r = root.getOrientation()
        yaw_now = math.atan2(r[3], r[0])
        # GPS offset from the robot root, in the robot's own frame
        dx, dy = gps[0] - root_pos[0], gps[1] - root_pos[1]
        off_x = math.cos(yaw_now) * dx + math.sin(yaw_now) * dy
        off_y = -math.sin(yaw_now) * dx + math.cos(yaw_now) * dy

        yaw = _yaw(pose.orientation)
        gps_x = self.__gps_origin[0] + pose.position.x
        gps_y = self.__gps_origin[1] + pose.position.y
        x = gps_x - (math.cos(yaw) * off_x - math.sin(yaw) * off_y)
        y = gps_y - (math.sin(yaw) * off_x + math.cos(yaw) * off_y)

        root.getField('translation').setSFVec3f([x, y, root_pos[2] + 0.05])
        root.getField('rotation').setSFRotation([0.0, 0.0, 1.0, yaw])
        root.resetPhysics()
        self.__cruising_speed_kmh = 0.0
        self.__steer = 0.0
        self.__driver.setCruisingSpeed(0.0)

    def __announce_reseed(self, seed):
        """Tell the referee (DEF REFEREE, customData) that the barrels go away, before removing them.

        The referee steps in parallel with this controller; if the barrels were removed in the same step it
        would read the removed nodes ("node not found" warnings, junk positions). -> steps to wait.
        """
        referee = self.__driver.getFromDef('REFEREE')
        if referee is None:
            return 0
        referee.getField('customData').setSFString(f'barrels {seed} {time.monotonic():.3f}')
        return 3

    def __reseed_barrels(self, seed):
        """Replace DEF BARRELS by a RobocrossBarrels with another randomSeed (same other fields).

        Remove + import, not setSFInt32 on the live node: Webots R2025a segfaulted when the
        field was changed after a supervisor had read the barrels inside the PROTO; remove()
        drops those references properly (tools/place_barrels.py). The world must declare the
        PROTO as IMPORTABLE EXTERNPROTO.
        """
        node = self.__driver.getFromDef('BARRELS')
        if node is None:
            self.__node.get_logger().warn('no DEF BARRELS RobocrossBarrels in the world')
            return
        task = node.getField('task').getSFString()
        count = node.getField('count').getSFInt32()
        p_pair = node.getField('pPair').getSFFloat()
        node.remove()
        self.__driver.getRoot().getField('children').importMFNodeFromString(
            -1, f'DEF BARRELS RobocrossBarrels {{ randomSeed {seed} task "{task}" count {count} pPair {p_pair} }}')
        self.__node.get_logger().info(f'barrels: new layout, randomSeed {seed}')

    def __base_link_transform(self):
        """odom->base_link from the GPS position and Xsens IMU orientation.

        Approximation: uses the GPS antenna's own position directly as
        base_link's, ignoring the ~1.6m lever arm between them (gps_link vs.
        base_link in vesta.urdf) - fine for visualization, not for precision
        localization. odom's origin is the vehicle's first GPS fix, so it
        starts near (0, 0, 0) like a normal odom frame rather than wherever
        the vehicle happens to spawn in the world.
        """
        t = TransformStamped()
        t.header.stamp = self.__sim_stamp()
        t.header.frame_id = ODOM_FRAME
        t.child_frame_id = BASE_FRAME
        t.transform.rotation = self.__orientation

        position = self.__gps.getValues()
        if not any(math.isnan(v) for v in position):
            if self.__gps_origin is None:
                self.__gps_origin = position
            t.transform.translation.x = position[0] - self.__gps_origin[0]
            t.transform.translation.y = position[1] - self.__gps_origin[1]
            t.transform.translation.z = position[2] - self.__gps_origin[2]
        return t

    def __radar_point_cloud(self):
        header = Header(frame_id=RADAR_FRAME, stamp=self.__sim_stamp())
        points = [
            # Webots' azimuth sign is flipped from the usual math convention: per
            # WbRadar.cpp, a target on the sensor's local +y side gets a NEGATIVE
            # azimuth. Negating sin(azimuth) here undoes that so +y in the cloud
            # really is the target's left, matching the frame's own +y=left axis.
            (target.distance * math.cos(target.azimuth), -target.distance * math.sin(target.azimuth), 0.0)
            for target in self.__radar.getTargets()
        ]
        return point_cloud2.create_cloud_xyz32(header, points)

    def __on_point_cloud(self, data):
        try:
            p = data
            p.header.stamp = self.__sim_stamp()
            p.header.frame_id = 'lidar_link'
            self.__lidar_publisher.publish(p)
        except  Exception as err:
            self.__node._logger.error(''.join(traceback.TracebackException.from_exception(err).format()))

    def __joint_states(self):
        steer = self.__driver.getSteeringAngle()* -1
        msg = JointState()
        msg.header.stamp = self.__sim_stamp()
        msg.name = list(STEER_JOINTS) + [joint_name for _, joint_name in self.__wheel_sensors]
        msg.position = [steer] * len(STEER_JOINTS) + [sensor.getValue() for sensor, _ in self.__wheel_sensors]
        return msg

