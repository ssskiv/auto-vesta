// Webots-only adapter so the real-car localization stack (navsat_transform + EKFs) runs unchanged.
// C++ port of offroad_bringup/sim_sensor_adapter.py.
//
//  * /gps/fix (Webots: PointStamped, world metres of the antenna) -> /gps/fix_ll (NavSatFix). The antenna is moved to
//    base_link with the IMU attitude and the gps_link TF (static lever arm, looked up once). The datum is the map origin,
//    i.e. world (2.2, -1.6). Latitude/longitude of the datum is a placeholder: TODO(user) real RoboCross venue.
//  * /imu/data (sim-stamped republish of /vesta/imu/data, covariances zero) -> /imu/data_cov with covariances filled in.

#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/nav_sat_status.hpp>
#include <tf2/exceptions.h>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace {
constexpr double kREarth = 6378137.0;
}  // namespace

class SimSensorAdapter : public rclcpp::Node
{
public:
  SimSensorAdapter()
  : Node("sim_sensor_adapter")
  {
    lat0_ = declare_parameter<double>("datum_lat", 54.3282);  // TODO(user)
    lon0_ = declare_parameter<double>("datum_lon", 48.3866);  // TODO(user)
    const auto off = declare_parameter<std::vector<double>>("map_origin_in_world", {2.2, -1.6});
    if (off.size() != 2) {
      throw std::invalid_argument("map_origin_in_world must have 2 elements");
    }
    off_x_ = off[0];
    off_y_ = off[1];
    sigma_xy_ = declare_parameter<double>("gps_sigma_xy", 0.3);

    buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    listener_ = std::make_shared<tf2_ros::TransformListener>(*buffer_, this);

    const auto reliable_qos = rclcpp::QoS(rclcpp::KeepLast(5)).reliable();
    pub_ll_ = create_publisher<sensor_msgs::msg::NavSatFix>("/gps/fix_ll", reliable_qos);
    pub_imu_ = create_publisher<sensor_msgs::msg::Imu>("/imu/data_cov", reliable_qos);

    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
      "/imu/data", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::Imu::ConstSharedPtr msg) {on_imu(msg);});
    sub_gps_ = create_subscription<geometry_msgs::msg::PointStamped>(
      "/gps/fix", rclcpp::SensorDataQoS(),
      [this](geometry_msgs::msg::PointStamped::ConstSharedPtr msg) {on_gps(msg);});
  }

private:
  void on_imu(const sensor_msgs::msg::Imu::ConstSharedPtr & msg)
  {
    const auto & o = msg->orientation;
    q_ = Eigen::Quaterniond(o.w, o.x, o.y, o.z);

    auto out = std::make_unique<sensor_msgs::msg::Imu>(*msg);
    for (size_t i : {0u, 4u, 8u}) {
      out->orientation_covariance[i] = 1e-4;
      out->angular_velocity_covariance[i] = 1e-4;
      out->linear_acceleration_covariance[i] = 1e-2;
    }
    pub_imu_->publish(std::move(out));
  }

  void on_gps(const geometry_msgs::msg::PointStamped::ConstSharedPtr & msg)
  {
    if (!q_) {
      return;
    }
    if (!lever_) {
      const std::string frame = msg->header.frame_id.empty() ? "gps_link" : msg->header.frame_id;
      try {
        const auto t = buffer_->lookupTransform("base_link", frame, tf2::TimePointZero);
        const auto & tr = t.transform.translation;
        lever_ = Eigen::Vector3d(tr.x, tr.y, tr.z);
      } catch (const tf2::TransformException &) {
        return;
      }
    }

    // base_link position in world: antenna minus rotated lever arm
    const Eigen::Vector3d antenna(msg->point.x, msg->point.y, msg->point.z);
    const Eigen::Vector3d p = antenna - q_->toRotationMatrix() * (*lever_);
    const double east = p.x() - off_x_;   // map metres
    const double north = p.y() - off_y_;

    auto f = std::make_unique<sensor_msgs::msg::NavSatFix>();
    f->header.stamp = now();
    f->header.frame_id = "base_link";
    f->status.status = sensor_msgs::msg::NavSatStatus::STATUS_FIX;
    f->status.service = sensor_msgs::msg::NavSatStatus::SERVICE_GPS;
    f->latitude = lat0_ + rad2deg(north / kREarth);
    f->longitude = lon0_ + rad2deg(east / (kREarth * std::cos(deg2rad(lat0_))));
    f->altitude = p.z();
    const double var_xy = sigma_xy_ * sigma_xy_;
    f->position_covariance = {var_xy, 0.0, 0.0, 0.0, var_xy, 0.0, 0.0, 0.0, 1.0};
    f->position_covariance_type = sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_DIAGONAL_KNOWN;
    pub_ll_->publish(std::move(f));
  }

  static double deg2rad(double d) {return d * M_PI / 180.0;}
  static double rad2deg(double r) {return r * 180.0 / M_PI;}

  double lat0_{0.0};
  double lon0_{0.0};
  double off_x_{0.0};
  double off_y_{0.0};
  double sigma_xy_{0.3};
  std::optional<Eigen::Quaterniond> q_;
  std::optional<Eigen::Vector3d> lever_;

  std::shared_ptr<tf2_ros::Buffer> buffer_;
  std::shared_ptr<tf2_ros::TransformListener> listener_;
  rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr pub_ll_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_imu_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_gps_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimSensorAdapter>());
  rclcpp::shutdown();
  return 0;
}
