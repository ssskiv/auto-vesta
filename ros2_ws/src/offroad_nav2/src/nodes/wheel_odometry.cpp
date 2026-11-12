// C++ port of offroad_vehicle_interface/wheel_odometry.py.
// /joint_states (rear wheel angles) -> /wheel/odometry (twist.linear.x only), for the local EKF.

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include "nav_msgs/msg/odometry.hpp"
#include "offroad_nav2/vehicle_logic.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

namespace {

constexpr std::array<const char*, 2> kRear = {"rear_left_wheel_joint", "rear_right_wheel_joint"};

class WheelOdometry : public rclcpp::Node {
 public:
  WheelOdometry() : Node("wheel_odometry") {
    const double radius = declare_parameter<double>("wheel_radius", 0.315);
    const double sign = declare_parameter<double>("sign", 1.0);  // TODO: check sign convention per sim/car
    ws_ = offroad::WheelSpeed(radius, sign);
    sigma_ = declare_parameter<double>("sigma_v", 0.1);

    pub_ = create_publisher<nav_msgs::msg::Odometry>("/wheel/odometry", rclcpp::QoS(10));
    sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", rclcpp::QoS(10),
        [this](sensor_msgs::msg::JointState::ConstSharedPtr m) { on_joint_states(*m); });
  }

 private:
  void on_joint_states(const sensor_msgs::msg::JointState& m) {
    std::vector<double> angles;
    angles.reserve(kRear.size());
    for (const char* name : kRear) {
      const auto it = std::find(m.name.begin(), m.name.end(), name);
      if (it == m.name.end()) {
        return;  // Python: ValueError from list.index -> ignore message
      }
      const auto idx = static_cast<std::size_t>(it - m.name.begin());
      if (idx >= m.position.size()) {
        return;  // Python would raise IndexError here; the C++ node drops the message instead
      }
      angles.push_back(m.position[idx]);
    }

    const double t = static_cast<double>(m.header.stamp.sec) + static_cast<double>(m.header.stamp.nanosec) * 1e-9;
    const std::optional<double> v = ws_.update(angles, t);
    if (!v) {
      return;
    }

    auto o = std::make_unique<nav_msgs::msg::Odometry>();
    o->header.stamp = m.header.stamp;
    o->header.frame_id = "odom";
    o->child_frame_id = "base_link";
    o->twist.twist.linear.x = *v;
    o->twist.covariance[0] = sigma_ * sigma_;
    for (std::size_t i : {7u, 14u, 21u, 28u, 35u}) {
      o->twist.covariance[i] = 1e3;
    }
    o->pose.covariance[0] = 1e3;
    pub_->publish(std::move(o));
  }

  offroad::WheelSpeed ws_{0.315};
  double sigma_ = 0.1;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WheelOdometry>());
  rclcpp::shutdown();
  return 0;
}
