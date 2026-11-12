// Virtual stop wall at the stop line, published as a PointCloud2 for nav2_collision_monitor (option B of the guide).
// C++ port of offroad_perception/stop_wall_node.py.
//
// Active only while the light is not GREEN (RED, YELLOW, or UNKNOWN) AND the car is approaching the line closer than
// `active_distance`. Always publishes (an empty cloud when inactive) because the collision monitor treats a silent
// source as unsafe. The stop line pose is a parameter: TODO(user) / world measurement (see traffic-light notes in README).

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <offroad_nav2/msg/traffic_light_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <tf2/exceptions.h>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

using TrafficLightState = offroad_nav2::msg::TrafficLightState;

class StopWall : public rclcpp::Node
{
public:
  StopWall()
  : Node("stop_wall")
  {
    const auto line = declare_parameter<std::vector<double>>("line_xy", {0.0, 0.0});
    if (line.size() != 2) {
      throw std::invalid_argument("line_xy must have 2 elements");
    }
    line_x_ = line[0];
    line_y_ = line[1];
    yaw_ = declare_parameter<double>("line_yaw", 0.0);
    width_ = declare_parameter<double>("width", 10.0);
    active_dist_ = declare_parameter<double>("active_distance", 25.0);
    offset_ = declare_parameter<double>("wall_offset", 4.0);   // wall this far BEFORE the line
    enabled_ = declare_parameter<bool>("enabled", false);

    buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    listener_ = std::make_shared<tf2_ros::TransformListener>(*buffer_, this);

    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/perception/stop_wall", rclcpp::QoS(5));
    sub_ = create_subscription<TrafficLightState>(
      "/perception/traffic_light", rclcpp::QoS(5),
      [this](TrafficLightState::ConstSharedPtr msg) {state_ = msg->state;});
    timer_ = create_wall_timer(std::chrono::milliseconds(100), [this]() {tick();});
  }

private:
  // Wall points as (x, y, z) triples, empty when the wall is not active.
  std::vector<float> wall_points() const
  {
    std::vector<float> pts;
    if (!enabled_ || state_ == TrafficLightState::GREEN) {
      return pts;
    }
    try {
      const auto t = buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
      const auto & tr = t.transform.translation;
      const double fx = std::cos(yaw_);
      const double fy = std::sin(yaw_);
      const double lx = line_x_ - offset_ * fx;
      const double ly = line_y_ - offset_ * fy;
      const double along = (lx - tr.x) * fx + (ly - tr.y) * fy;   // distance still to go to the wall
      if (0.0 < along && along < active_dist_) {
        // Dense sampling (0.1 m): the monitor needs >= min_points inside its polygon.
        // Same sample set as numpy.arange(-w/2, w/2 + 1e-6, 0.1).
        const double start = -width_ / 2.0;
        const double stop = width_ / 2.0 + 1e-6;
        const double step = 0.1;
        const auto n = static_cast<std::size_t>(std::ceil((stop - start) / step));
        pts.reserve(n * 3);
        for (std::size_t i = 0; i < n; ++i) {
          const double s = start + static_cast<double>(i) * step;
          pts.push_back(static_cast<float>(lx - s * fy));
          pts.push_back(static_cast<float>(ly + s * fx));
          pts.push_back(0.5f);
        }
      }
    } catch (const tf2::TransformException &) {
      pts.clear();
    }
    return pts;
  }

  void tick()
  {
    const std::vector<float> pts = wall_points();
    const std::size_t n = pts.size() / 3;

    auto m = std::make_unique<sensor_msgs::msg::PointCloud2>();
    m->header.stamp = now();
    m->header.frame_id = "map";
    m->height = 1;
    m->width = static_cast<uint32_t>(n);
    const char * names[3] = {"x", "y", "z"};
    m->fields.resize(3);
    for (uint32_t i = 0; i < 3; ++i) {
      m->fields[i].name = names[i];
      m->fields[i].offset = 4 * i;
      m->fields[i].datatype = sensor_msgs::msg::PointField::FLOAT32;
      m->fields[i].count = 1;
    }
    m->is_bigendian = false;
    m->point_step = 12;
    m->is_dense = true;
    m->row_step = static_cast<uint32_t>(12 * n);
    m->data.resize(12 * n);
    if (n > 0) {
      std::memcpy(m->data.data(), pts.data(), 12 * n);
    }
    pub_->publish(std::move(m));
  }

  double line_x_{0.0};
  double line_y_{0.0};
  double yaw_{0.0};
  double width_{10.0};
  double active_dist_{25.0};
  double offset_{4.0};
  bool enabled_{false};
  uint8_t state_{TrafficLightState::UNKNOWN};

  std::shared_ptr<tf2_ros::Buffer> buffer_;
  std::shared_ptr<tf2_ros::TransformListener> listener_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<TrafficLightState>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StopWall>());
  rclcpp::shutdown();
  return 0;
}
