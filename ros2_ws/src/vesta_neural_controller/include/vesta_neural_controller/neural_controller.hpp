// Nav2 controller plugin, Yandex-rover style: the RL policy (ONNX) and a
// classical controller (RPP, loaded inside this plugin) both propose a
// command; each is rolled out on the local costmap and the better safe one
// is sent. Neither safe -> stop.
#ifndef VESTA_NEURAL_CONTROLLER__NEURAL_CONTROLLER_HPP_
#define VESTA_NEURAL_CONTROLLER__NEURAL_CONTROLLER_HPP_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav2_core/controller.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_costmap_2d/footprint_collision_checker.hpp"
#include "nav_msgs/msg/path.hpp"
#include "pluginlib/class_loader.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "rclcpp_lifecycle/lifecycle_publisher.hpp"
#include "std_msgs/msg/string.hpp"
#include "tf2_ros/buffer.h"
#include "visualization_msgs/msg/marker_array.hpp"

#include "vesta_neural_controller/observation.hpp"
#include "vesta_neural_controller/onnx_policy.hpp"

namespace vesta_neural_controller
{

struct Command
{
  double v = 0.0;
  double delta = 0.0;
};

struct Rollout
{
  std::vector<std::array<double, 3>> poses;  // x, y, yaw
  bool collides = false;
  double score = 0.0;
};

class NeuralController : public nav2_core::Controller
{
public:
  NeuralController() = default;
  ~NeuralController() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent, std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;
  void cleanup() override;
  void activate() override;
  void deactivate() override;
  void setPlan(const nav_msgs::msg::Path & path) override;
  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose, const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override;
  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

private:
  bool updateTrackerPoints();
  bool policyCommand(
    const geometry_msgs::msg::PoseStamped & pose, const geometry_msgs::msg::Twist & velocity,
    Command & cmd, double & s_now);
  Rollout rollout(double x, double y, double yaw, double v0, const Command & cmd, double s_now);
  void publishDebug(const std::string & source, const Rollout * nn, const Rollout & rpp, int chosen);

  rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
  rclcpp::Logger logger_{rclcpp::get_logger("NeuralController")};
  rclcpp::Clock::SharedPtr clock_;
  std::string name_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D * costmap_ = nullptr;
  std::unique_ptr<nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *>> collision_checker_;

  std::unique_ptr<pluginlib::ClassLoader<nav2_core::Controller>> fallback_loader_;
  nav2_core::Controller::Ptr fallback_;
  OnnxPolicy policy_;

  nav_msgs::msg::Path plan_;
  PathTracker tracker_;
  Command last_;

  // parameters
  bool use_nn_ = true;
  double max_speed_ = 5.0;          // the policy's training max speed (action / obs scale)
  double speed_limit_ = 5.0;        // effective cap (setSpeedLimit)
  double rollout_time_ = 2.0;
  double rollout_dt_ = 0.1;
  double nn_margin_ = 0.5;
  double cost_weight_ = 1.0;
  double tau_v_ = 0.6, accel_max_ = 2.5, tau_delta_ = 0.1, steer_rate_ = 1.0;
  double transform_tolerance_ = 0.2;

  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::String>> source_pub_;
  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<visualization_msgs::msg::MarkerArray>> markers_pub_;
};

}  // namespace vesta_neural_controller

#endif  // VESTA_NEURAL_CONTROLLER__NEURAL_CONTROLLER_HPP_
