// C++ port of offroad_vehicle_interface/cmd_to_ackermann_node.py.
// Real-car path: Twist -> steering angle + speed -> DbwAdapter (logging stub until decision 7.6). Not used in sim.

#include <chrono>
#include <memory>
#include <optional>
#include <string>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "offroad_nav2/dbw_adapter.hpp"
#include "offroad_nav2/vehicle_logic.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"

namespace {

constexpr double kVMin = 0.1;      // steer_from_twist default v_min
constexpr double kPeriodS = 0.05;  // timer period and initial dt

class CmdToAckermann : public rclcpp::Node {
 public:
  CmdToAckermann() : Node("cmd_to_ackermann") {
    wheelbase_ = declare_parameter<double>("wheelbase", 2.648);
    max_steer_ = declare_parameter<double>("max_steer", 0.6);
    max_rate_ = declare_parameter<double>("max_steer_rate", 0.5);  // rad/s, TODO(user): measure

    // TODO(user) decision 7.6: replace with the OSCC/ros2cco adapter.
    rclcpp::Logger logger = get_logger();
    adapter_ = std::make_unique<offroad::LoggingDbwAdapter>(
        [logger](offroad::LogLevel level, const std::string& msg) {
          if (level == offroad::LogLevel::Warning) {
            RCLCPP_WARN(logger, "%s", msg.c_str());
          } else {
            RCLCPP_INFO(logger, "%s", msg.c_str());
          }
        });

    pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/vehicle/ackermann_cmd", rclcpp::QoS(10));
    cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel_nav", rclcpp::QoS(10),
        [this](geometry_msgs::msg::Twist::ConstSharedPtr m) {
          wd_.update(m->linear.x, m->angular.z, now_s());
        });
    engage_sub_ = create_subscription<std_msgs::msg::Bool>(
        "/vehicle/engage", rclcpp::QoS(10),
        [this](std_msgs::msg::Bool::ConstSharedPtr m) { engaged_ = m->data; });
    estop_sub_ = create_subscription<std_msgs::msg::Bool>(
        "/vehicle/emergency_stop", rclcpp::QoS(10),
        [this](std_msgs::msg::Bool::ConstSharedPtr m) {
          if (m->data) {
            adapter_->emergency_stop();
          }
        });
    timer_ = rclcpp::create_timer(this, get_clock(), rclcpp::Duration::from_seconds(kPeriodS), [this]() { tick(); });
  }

 private:
  double now_s() {
    return get_clock()->now().nanoseconds() * 1e-9;
  }

  void tick() {
    const double t = now_s();
    const double dt = t_prev_ ? (t - *t_prev_) : kPeriodS;
    t_prev_ = t;

    const offroad::WatchdogOutput out = wd_.output(t);
    double v = out.v;
    const double w = out.w;
    if (!engaged_ || adapter_->status().estop) {
      v = 0.0;
    }
    steer_ = offroad::steer_from_twist(v, w, wheelbase_, steer_, max_steer_, kVMin, max_rate_, dt);
    adapter_->send(steer_, v, 0.0);

    auto msg = std::make_unique<ackermann_msgs::msg::AckermannDriveStamped>();
    const rclcpp::Time stamp = get_clock()->now();
    msg->header.stamp = stamp;
    msg->drive.steering_angle = steer_;
    msg->drive.speed = v;
    pub_->publish(std::move(msg));
  }

  double wheelbase_ = 2.648;
  double max_steer_ = 0.6;
  double max_rate_ = 0.5;
  bool engaged_ = false;
  offroad::ZeroClampWatchdog wd_{0.3};
  double steer_ = 0.0;
  std::optional<double> t_prev_;
  std::unique_ptr<offroad::DbwAdapter> adapter_;

  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr engage_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr estop_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CmdToAckermann>());
  rclcpp::shutdown();
  return 0;
}
