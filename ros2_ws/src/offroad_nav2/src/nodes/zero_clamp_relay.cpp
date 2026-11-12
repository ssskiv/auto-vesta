// C++ port of offroad_vehicle_interface/zero_clamp_relay.py.
// /cmd_vel_nav (Twist) -> /cmd_vel with linear.x >= 0 and a 0.3 s watchdog. Mandatory in sim: the sim adapter
// reverses on negative speed.

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>

#include "geometry_msgs/msg/twist.hpp"
#include "offroad_nav2/vehicle_logic.hpp"
#include "rclcpp/rclcpp.hpp"

namespace {

class ZeroClampRelay : public rclcpp::Node {
 public:
  ZeroClampRelay() : Node("zero_clamp_relay") {
    const double watchdog_s = declare_parameter<double>("watchdog_s", 0.3);
    wd_ = offroad::ZeroClampWatchdog(watchdog_s);
    const std::string out_topic = declare_parameter<std::string>("out_topic", "/cmd_vel");
    const std::string in_topic = declare_parameter<std::string>("in_topic", "/cmd_vel_nav");

    pub_ = create_publisher<geometry_msgs::msg::Twist>(out_topic, rclcpp::QoS(10));
    sub_ = create_subscription<geometry_msgs::msg::Twist>(
        in_topic, rclcpp::QoS(10),
        [this](geometry_msgs::msg::Twist::ConstSharedPtr m) { on_cmd(*m); });
    timer_ = rclcpp::create_timer(this, get_clock(), rclcpp::Duration::from_seconds(0.05), [this]() { tick(); });
  }

 private:
  double now_s() {
    return get_clock()->now().nanoseconds() * 1e-9;
  }

  void on_cmd(const geometry_msgs::msg::Twist& m) {
    const double t = now_s();
    const std::size_t n0 = wd_.clamped();
    wd_.update(m.linear.x, m.angular.z, t);
    if (wd_.clamped() != n0 && t - last_log_ > 2.0) {
      last_log_ = t;
      RCLCPP_WARN(get_logger(), "clamped negative linear.x=%.2f to 0 (%zu total)", m.linear.x, wd_.clamped());
    }
    emit(t);
  }

  void tick() {
    const double t = now_s();
    if (wd_.output(t).stale) {
      emit(t);
    }
  }

  void emit(double t) {
    const offroad::WatchdogOutput out = wd_.output(t);
    auto msg = std::make_unique<geometry_msgs::msg::Twist>();
    msg->linear.x = out.v;
    msg->angular.z = out.w;
    pub_->publish(std::move(msg));
  }

  offroad::ZeroClampWatchdog wd_{0.3};
  double last_log_ = 0.0;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ZeroClampRelay>());
  rclcpp::shutdown();
  return 0;
}
