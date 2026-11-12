// Traffic-light state publisher.
// C++ port of offroad_perception/traffic_light_node.py.
//
// NO detector or light-state classifier exists yet (decision 7.5). Until one does, this node runs in a clearly labelled
// STAND-IN mode that is NOT a perception result: it reads the light's state from /standin/traffic_light_state (world
// truth from Webots) and publishes /perception/traffic_light with confidence 0 and track_id -1 (stand-in marker).
// With source:=none it publishes UNKNOWN only. TODO(user): replace with a model-based node (P10).

#include <cctype>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <string>

#include <offroad_nav2/msg/traffic_light_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

using TrafficLightState = offroad_nav2::msg::TrafficLightState;

class TrafficLightNode : public rclcpp::Node
{
public:
  TrafficLightNode()
  : Node("traffic_light_node")
  {
    source_ = declare_parameter<std::string>("source", "none");   // none | standin
    pub_ = create_publisher<TrafficLightState>("/perception/traffic_light", rclcpp::QoS(5));
    sub_ = create_subscription<std_msgs::msg::String>(
      "/standin/traffic_light_state", rclcpp::QoS(5),
      [this](std_msgs::msg::String::ConstSharedPtr msg) {on_state(msg);});
    timer_ = create_wall_timer(std::chrono::milliseconds(100), [this]() {tick();});
    if (source_ == "standin") {
      RCLCPP_WARN(get_logger(), "STAND-IN traffic light state (world truth), not a perception result");
    }
  }

private:
  void on_state(const std_msgs::msg::String::ConstSharedPtr & msg)
  {
    std::string key = msg->data;
    for (auto & c : key) {
      c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    if (key == "RED") {
      state_ = TrafficLightState::RED;
    } else if (key == "YELLOW" || key == "ORANGE") {
      state_ = TrafficLightState::YELLOW;
    } else if (key == "GREEN") {
      state_ = TrafficLightState::GREEN;
    } else {
      state_ = TrafficLightState::UNKNOWN;
    }
  }

  void tick()
  {
    auto m = std::make_unique<TrafficLightState>();
    m->header.stamp = now();
    m->header.frame_id = "map";
    m->state = (source_ == "standin") ? state_ : TrafficLightState::UNKNOWN;
    m->confidence = 0.0f;
    m->distance_m = std::numeric_limits<float>::quiet_NaN();
    m->track_id = -1;
    pub_->publish(std::move(m));
  }

  std::string source_;
  uint8_t state_{TrafficLightState::UNKNOWN};
  rclcpp::Publisher<TrafficLightState>::SharedPtr pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TrafficLightNode>());
  rclcpp::shutdown();
  return 0;
}
