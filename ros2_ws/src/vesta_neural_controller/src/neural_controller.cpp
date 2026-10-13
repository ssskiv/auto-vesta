#include "vesta_neural_controller/neural_controller.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>

#include "nav2_core/exceptions.hpp"
#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2/utils.h"

using nav2_util::declare_parameter_if_not_declared;

namespace vesta_neural_controller
{

namespace
{
double yawOf(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}
}  // namespace

void NeuralController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent, std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf, std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent;
  auto node = parent.lock();
  if (!node) {
    throw nav2_core::PlannerException("Unable to lock node");
  }
  name_ = name;
  tf_ = tf;
  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros->getCostmap();
  logger_ = node->get_logger();
  clock_ = node->get_clock();

  auto param = [&](const std::string & key, const rclcpp::ParameterValue & def) {
      declare_parameter_if_not_declared(node, name_ + "." + key, def);
      return node->get_parameter(name_ + "." + key);
    };
  const std::string model_path = param("model_path", rclcpp::ParameterValue(std::string(""))).as_string();
  const std::string fallback_plugin = param(
    "fallback_plugin",
    rclcpp::ParameterValue(std::string(
      "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"))).as_string();
  use_nn_ = param("use_nn", rclcpp::ParameterValue(true)).as_bool();
  max_speed_ = param("max_speed", rclcpp::ParameterValue(5.0)).as_double();
  rollout_time_ = param("rollout_time", rclcpp::ParameterValue(2.0)).as_double();
  rollout_dt_ = param("rollout_dt", rclcpp::ParameterValue(0.1)).as_double();
  nn_margin_ = param("nn_margin", rclcpp::ParameterValue(0.5)).as_double();
  cost_weight_ = param("cost_weight", rclcpp::ParameterValue(1.0)).as_double();
  tau_v_ = param("tau_v", rclcpp::ParameterValue(0.6)).as_double();
  accel_max_ = param("accel_max", rclcpp::ParameterValue(2.5)).as_double();
  tau_delta_ = param("tau_delta", rclcpp::ParameterValue(0.1)).as_double();
  steer_rate_ = param("steer_rate", rclcpp::ParameterValue(1.0)).as_double();
  transform_tolerance_ = param("transform_tolerance", rclcpp::ParameterValue(0.2)).as_double();
  speed_limit_ = max_speed_;

  // The classical planner runs inside this plugin, with its parameters under <name>.fallback.*
  fallback_loader_ = std::make_unique<pluginlib::ClassLoader<nav2_core::Controller>>(
    "nav2_core", "nav2_core::Controller");
  fallback_ = fallback_loader_->createSharedInstance(fallback_plugin);
  fallback_->configure(parent, name_ + ".fallback", tf, costmap_ros);

  if (use_nn_) {
    std::string error;
    if (model_path.empty()) {
      RCLCPP_WARN(logger_, "%s: model_path is empty, driving with the fallback only", name_.c_str());
    } else if (!policy_.load(model_path, error)) {
      RCLCPP_ERROR(logger_, "%s: cannot load %s (%s), driving with the fallback only",
        name_.c_str(), model_path.c_str(), error.c_str());
    } else {
      RCLCPP_INFO(logger_, "%s: loaded policy %s", name_.c_str(), model_path.c_str());
    }
  }

  collision_checker_ = std::make_unique<
    nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *>>(costmap_);
  source_pub_ = node->create_publisher<std_msgs::msg::String>("neural_controller/source", 10);
  markers_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>("neural_controller/rollouts", 1);
}

void NeuralController::cleanup()
{
  fallback_->cleanup();
  source_pub_.reset();
  markers_pub_.reset();
}

void NeuralController::activate()
{
  fallback_->activate();
  source_pub_->on_activate();
  markers_pub_->on_activate();
}

void NeuralController::deactivate()
{
  fallback_->deactivate();
  source_pub_->on_deactivate();
  markers_pub_->on_deactivate();
}

void NeuralController::setPlan(const nav_msgs::msg::Path & path)
{
  plan_ = path;
  tracker_ = PathTracker();
  fallback_->setPlan(path);
}

void NeuralController::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  if (speed_limit == 0.0) {  // nav2_costmap_2d::NO_SPEED_LIMIT
    speed_limit_ = max_speed_;
  } else if (percentage) {
    speed_limit_ = max_speed_ * speed_limit / 100.0;
  } else {
    speed_limit_ = std::min(max_speed_, speed_limit);
  }
  fallback_->setSpeedLimit(speed_limit, percentage);
}

bool NeuralController::updateTrackerPoints()
{
  if (plan_.poses.empty()) {
    return false;
  }
  geometry_msgs::msg::TransformStamped t;
  try {
    t = tf_->lookupTransform(
      costmap_ros_->getGlobalFrameID(), plan_.header.frame_id, tf2::TimePointZero,
      tf2::durationFromSec(transform_tolerance_));
  } catch (const tf2::TransformException & e) {
    RCLCPP_WARN_THROTTLE(logger_, *clock_, 2000, "%s: %s", name_.c_str(), e.what());
    return false;
  }
  const double tx = t.transform.translation.x, ty = t.transform.translation.y;
  const double yaw = yawOf(t.transform.rotation);
  const double c = std::cos(yaw), s = std::sin(yaw);
  std::vector<double> xy;
  xy.reserve(plan_.poses.size() * 2);
  for (const auto & p : plan_.poses) {
    const double x = p.pose.position.x, y = p.pose.position.y;
    xy.push_back(tx + c * x - s * y);
    xy.push_back(ty + s * x + c * y);
  }
  tracker_.setPoints(xy);
  return true;
}

bool NeuralController::policyCommand(
  const geometry_msgs::msg::PoseStamped & pose, const geometry_msgs::msg::Twist & velocity,
  Command & cmd, double & s_now)
{
  static thread_local std::vector<float> patch(obs::PATCH_SIZE * obs::PATCH_SIZE);
  std::array<float, obs::VEC_SIZE> vec;
  const double px = pose.pose.position.x, py = pose.pose.position.y;
  const double yaw = yawOf(pose.pose.orientation);
  const VecInfo info = buildVec(
    tracker_, px, py, yaw, velocity.linear.x, velocity.angular.z, last_.v, last_.delta, max_speed_, vec.data());
  s_now = info.s;

  const CostmapView view{
    costmap_->getCharMap(), static_cast<int>(costmap_->getSizeInCellsX()),
    static_cast<int>(costmap_->getSizeInCellsY()), costmap_->getOriginX(), costmap_->getOriginY(),
    costmap_->getResolution()};
  buildPatch(view, px, py, yaw, patch.data());

  float action[2];
  std::string error;
  if (!policy_.infer(patch.data(), vec.data(), action, error)) {
    RCLCPP_ERROR_THROTTLE(logger_, *clock_, 2000, "%s: inference failed: %s", name_.c_str(), error.c_str());
    return false;
  }
  // the speed action is a fraction of the path speed limit (curves, goal)
  actionToCommand(action[0], action[1], speedLimit(tracker_, info.s, max_speed_), cmd.v, cmd.delta);
  return true;
}

Rollout NeuralController::rollout(double x, double y, double yaw, double v0, const Command & cmd, double s_now)
{
  // Same first-order model as vesta_rl/sim2d.py Vehicle (nominal parameters).
  Rollout r;
  const int steps = std::max(1, static_cast<int>(std::round(rollout_time_ / rollout_dt_)));
  const int sub = 5;
  const double h = rollout_dt_ / sub;
  const auto footprint = costmap_ros_->getRobotFootprint();
  double v = v0, delta = last_.delta, cost_sum = 0.0;
  r.poses.push_back({x, y, yaw});
  for (int i = 0; i < steps; ++i) {
    for (int k = 0; k < sub; ++k) {
      const double dv = (cmd.v - v) * std::min(1.0, h / tau_v_);
      v += std::clamp(dv, -accel_max_ * h, accel_max_ * h);
      const double dd = (cmd.delta - delta) * std::min(1.0, h / tau_delta_);
      delta += std::clamp(dd, -steer_rate_ * h, steer_rate_ * h);
      x += v * std::cos(yaw) * h;
      y += v * std::sin(yaw) * h;
      yaw += v * std::tan(delta) / obs::WHEELBASE * h;
    }
    r.poses.push_back({x, y, yaw});
    const double cost = collision_checker_->footprintCostAtPose(x, y, yaw, footprint);
    if (cost >= static_cast<double>(nav2_costmap_2d::LETHAL_OBSTACLE) &&
      cost != static_cast<double>(nav2_costmap_2d::NO_INFORMATION))
    {
      r.collides = true;
      break;
    }
    cost_sum += cost == static_cast<double>(nav2_costmap_2d::NO_INFORMATION) ? 0.0 : cost;
  }
  const double progress = tracker_.empty() ? 0.0 : tracker_.peek(x, y).s - s_now;
  r.score = progress - cost_weight_ * cost_sum / (steps * static_cast<double>(obs::LETHAL));
  return r;
}

geometry_msgs::msg::TwistStamped NeuralController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose, const geometry_msgs::msg::Twist & velocity,
  nav2_core::GoalChecker * goal_checker)
{
  std::lock_guard<nav2_costmap_2d::Costmap2D::mutex_t> lock(*costmap_->getMutex());
  const bool have_path = updateTrackerPoints();

  // 1. classical proposal
  Command rpp;
  bool rpp_ok = true;
  try {
    const auto tw = fallback_->computeVelocityCommands(pose, velocity, goal_checker);
    rpp.v = tw.twist.linear.x;
    rpp.delta = std::abs(rpp.v) > 1e-3 ?
      std::clamp(std::atan(tw.twist.angular.z * obs::WHEELBASE / rpp.v), -obs::MAX_STEER, obs::MAX_STEER) :
      last_.delta;
  } catch (const std::exception & e) {
    rpp_ok = false;
    RCLCPP_WARN_THROTTLE(logger_, *clock_, 2000, "%s: fallback failed: %s", name_.c_str(), e.what());
  }
  rateLimit(rpp.v, rpp.delta, last_.v, last_.delta);
  rpp.v = std::min(rpp.v, speed_limit_);

  // 2. neural proposal
  Command nn;
  double s_now = 0.0;
  bool nn_ok = false;
  if (have_path && use_nn_ && policy_.loaded()) {
    nn_ok = policyCommand(pose, velocity, nn, s_now);
    rateLimit(nn.v, nn.delta, last_.v, last_.delta);
    nn.v = std::min(nn.v, speed_limit_);
  } else if (have_path) {
    s_now = tracker_.project(pose.pose.position.x, pose.pose.position.y).s;
  }

  // 3. roll out both, pick the better safe one
  const double x = pose.pose.position.x, y = pose.pose.position.y, yaw = yawOf(pose.pose.orientation);
  const Rollout rpp_r = rollout(x, y, yaw, velocity.linear.x, rpp, s_now);
  Rollout nn_r;
  if (nn_ok) {
    nn_r = rollout(x, y, yaw, velocity.linear.x, nn, s_now);
  }
  const bool rpp_safe = rpp_ok && !rpp_r.collides;
  const bool nn_safe = nn_ok && !nn_r.collides;

  Command out;
  std::string source;
  int chosen;
  if (nn_safe && (!rpp_safe || nn_r.score >= rpp_r.score - nn_margin_)) {
    out = nn;
    source = "NN";
    chosen = 0;
  } else if (rpp_safe) {
    out = rpp;
    source = "RPP";
    chosen = 1;
  } else {
    out.v = 0.0;
    out.delta = last_.delta;
    source = "STOP";
    chosen = 2;
  }
  last_ = out;
  publishDebug(source, nn_ok ? &nn_r : nullptr, rpp_r, chosen);

  geometry_msgs::msg::TwistStamped cmd;
  cmd.header.frame_id = pose.header.frame_id;
  cmd.header.stamp = clock_->now();
  cmd.twist.linear.x = out.v;
  cmd.twist.angular.z = out.v * std::tan(out.delta) / obs::WHEELBASE;
  return cmd;
}

void NeuralController::publishDebug(
  const std::string & source, const Rollout * nn, const Rollout & rpp, int chosen)
{
  std_msgs::msg::String s;
  s.data = source;
  source_pub_->publish(s);

  if (markers_pub_->get_subscription_count() == 0) {
    return;
  }
  visualization_msgs::msg::MarkerArray arr;
  auto add = [&](const Rollout & r, const std::string & ns, bool is_chosen) {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = costmap_ros_->getGlobalFrameID();
      m.header.stamp = clock_->now();
      m.ns = ns;
      m.id = 0;
      m.type = visualization_msgs::msg::Marker::LINE_STRIP;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.orientation.w = 1.0;
      m.scale.x = is_chosen ? 0.15 : 0.08;
      m.color.a = 1.0;
      if (r.collides) {
        m.color.r = 1.0;
      } else if (is_chosen) {
        m.color.g = 1.0;
      } else {
        m.color.r = m.color.g = m.color.b = 0.6;
      }
      for (const auto & p : r.poses) {
        geometry_msgs::msg::Point pt;
        pt.x = p[0];
        pt.y = p[1];
        pt.z = 0.1;
        m.points.push_back(pt);
      }
      arr.markers.push_back(m);
    };
  if (nn) {
    add(*nn, "nn", chosen == 0);
  }
  add(rpp, "rpp", chosen == 1);
  markers_pub_->publish(arr);
}

}  // namespace vesta_neural_controller

PLUGINLIB_EXPORT_CLASS(vesta_neural_controller::NeuralController, nav2_core::Controller)
