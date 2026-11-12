"""Traffic-light state publisher.

NO detector or light-state classifier exists yet (decision 7.5). Until one does, this node runs in a clearly labelled
STAND-IN mode that is NOT a perception result: it reads the light's known position (map frame) and, from the Webots
world, its state, and publishes /perception/traffic_light with confidence 0 and track_id -1 (stand-in marker).
With source:=none it publishes UNKNOWN only. TODO(user): replace with a model-based node (P10).
"""
import math

import rclpy
from offroad_nav2.msg import TrafficLightState
from rclpy.node import Node
from std_msgs.msg import String


class TrafficLightNode(Node):
    def __init__(self):
        super().__init__('traffic_light_node')
        self.source = self.declare_parameter('source', 'none').value      # none | standin
        self.pub = self.create_publisher(TrafficLightState, '/perception/traffic_light', 5)
        self.state = TrafficLightState.UNKNOWN
        self.create_subscription(String, '/standin/traffic_light_state', self.on_state, 5)
        self.create_timer(0.1, self.tick)
        if self.source == 'standin':
            self.get_logger().warn('STAND-IN traffic light state (world truth), not a perception result')

    def on_state(self, m):
        self.state = {'RED': 1, 'YELLOW': 2, 'ORANGE': 2, 'GREEN': 3}.get(m.data.upper(), 0)

    def tick(self):
        m = TrafficLightState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.state = self.state if self.source == 'standin' else TrafficLightState.UNKNOWN
        m.confidence, m.distance_m, m.track_id = 0.0, math.nan, -1
        self.pub.publish(m)


def main():
    rclpy.init()
    rclpy.spin(TrafficLightNode())


if __name__ == '__main__':
    main()
