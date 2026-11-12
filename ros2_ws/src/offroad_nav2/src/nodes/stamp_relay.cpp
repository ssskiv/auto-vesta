// C++ port of offroad_perception/stamp_relay.py (Webots-only).
//
// Webots camera plugins publish /stereo_raw/{left,right}/image_color with wall-clock stamps. This node re-stamps each
// left/right pair with the node clock (sim time when use_sim_time is set), republishes the left image and both
// camera_infos, and publishes row-cropped copies for the stereo matcher.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "builtin_interfaces/msg/time.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace {

using builtin_interfaces::msg::Time;
using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::Image;

constexpr size_t kMaxPairs = 32;   // pair map is trimmed when it grows beyond this
constexpr size_t kKeepPairs = 16;  // ... down to the newest this many entries
constexpr int kLeft = 0;
constexpr int kRight = 1;
constexpr std::array<const char *, 2> kSides = {"left", "right"};

int64_t key_of(const Time & t)
{
  return static_cast<int64_t>(t.sec) * 1000000000LL + static_cast<int64_t>(t.nanosec);
}

Time to_time_msg(const rclcpp::Time & t)
{
  const int64_t ns = t.nanoseconds();
  Time m;
  m.sec = static_cast<int32_t>(ns / 1000000000LL);
  m.nanosec = static_cast<uint32_t>(ns % 1000000000LL);
  return m;
}

struct PairEntry
{
  Time sim_stamp;
  bool skip;  // true: this pair arrived within min_period of the last published one
};

class StampRelay : public rclcpp::Node
{
public:
  StampRelay()
  : Node("stamp_relay")
  {
    baseline_ = declare_parameter<double>("baseline", 0.20);
    in_ns_ = declare_parameter<std::string>("in_ns", "/stereo_raw");
    suffix_ = declare_parameter<std::string>("in_image_suffix", "image_color");
    crop_top_ = declare_parameter<int64_t>("crop_top", 340);
    crop_bottom_ = declare_parameter<int64_t>("crop_bottom", 624);
    pair_tol_ns_ = static_cast<int64_t>(declare_parameter<double>("pair_tolerance_ms", 8.0) * 1e6);
    min_period_ = 1.0 / declare_parameter<double>("max_rate_hz", 10.0);

    const auto out_qos = rclcpp::QoS(5).reliable();
    // best-effort drops fragments of the 3.7 MB images over UDP; Webots publishes reliable
    const auto in_qos = rclcpp::QoS(3).reliable();

    for (int i = 0; i < 2; ++i) {
      const std::string side = kSides[i];
      pub_img_[i] = create_publisher<Image>("/stereo/" + side + "/image_rect", out_qos);
      pub_info_[i] = create_publisher<CameraInfo>("/stereo/" + side + "/camera_info", out_qos);
      cpub_img_[i] = create_publisher<Image>("/stereo_crop/" + side + "/image_rect", out_qos);
      cpub_info_[i] = create_publisher<CameraInfo>("/stereo_crop/" + side + "/camera_info", out_qos);

      sub_img_[i] = create_subscription<Image>(
        in_ns_ + "/" + side + "/" + suffix_, in_qos,
        [this, i](Image::ConstSharedPtr m) {on_image(m, i);});
      sub_info_[i] = create_subscription<CameraInfo>(
        in_ns_ + "/" + side + "/camera_info", rclcpp::SensorDataQoS(),
        [this, i](CameraInfo::ConstSharedPtr m) {
          k_[i] = m->k;
          have_k_[i] = true;
        });
    }

    RCLCPP_INFO(
      get_logger(), "restamping with sim time (use_sim_time=%s)",
      get_parameter("use_sim_time").as_bool() ? "True" : "False");
  }

private:
  // Returns the sim stamp and skip flag for the pair that contains `orig`, creating the entry on first sight.
  PairEntry lookup(int64_t key)
  {
    for (const auto & [k, e] : pairs_) {
      // left/right of one Webots step carry wall stamps a few ms apart: treat them as one pair
      if (std::llabs(k - key) < pair_tol_ns_) {
        return e;
      }
    }
    const rclcpp::Time now = this->now();
    const double t = static_cast<double>(now.nanoseconds()) * 1e-9;
    PairEntry e{to_time_msg(now), (t - last_pub_) < min_period_ * 0.9};
    if (!e.skip) {
      last_pub_ = t;
    }
    pairs_.emplace(key, e);
    if (pairs_.size() > kMaxPairs) {
      pairs_.erase(pairs_.begin(), std::next(pairs_.begin(), pairs_.size() - kKeepPairs));
    }
    return e;
  }

  void on_image(const Image::ConstSharedPtr & msg, int i)
  {
    if (!have_k_[i]) {
      return;  // wait for the first camera_info
    }
    const size_t h = msg->height;
    const size_t step = msg->step;
    if (msg->data.size() < h * step) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "short image on %s, dropped", kSides[i]);
      return;
    }
    const std::array<double, 9> K = k_[i];
    const PairEntry entry = lookup(key_of(msg->header.stamp));
    if (entry.skip) {
      return;
    }
    const Time & stamp = entry.sim_stamp;

    // camera_info (built once, used for the republished and the cropped copy)
    auto info = std::make_unique<CameraInfo>();
    info->header = msg->header;
    info->header.stamp = stamp;
    info->width = msg->width;
    info->height = msg->height;
    info->distortion_model = "plumb_bob";
    info->d.assign(5, 0.0);
    info->k = K;
    info->r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const double tx = (i == kRight) ? -K[0] * baseline_ : 0.0;
    info->p = {K[0], 0.0, K[2], tx, 0.0, K[4], K[5], 0.0, 0.0, 0.0, 1.0, 0.0};

    if (i == kLeft) {
      // the only full-frame copy: the republished image needs the sim stamp
      auto out = std::make_unique<Image>(*msg);
      out->header.stamp = stamp;
      pub_img_[i]->publish(std::move(out));
      pub_info_[i]->publish(std::make_unique<CameraInfo>(*info));
    }

    // cropped copy for the stereo matcher: rows [r0, r1) only
    const int64_t r0 = std::clamp<int64_t>(crop_top_, 0, static_cast<int64_t>(h));
    const int64_t r1 = std::clamp<int64_t>(crop_bottom_, r0, static_cast<int64_t>(h));
    if (r1 <= r0) {
      return;
    }
    const size_t rows = static_cast<size_t>(r1 - r0);

    auto cm = std::make_unique<Image>();
    cm->header = msg->header;
    cm->header.stamp = stamp;
    cm->height = static_cast<uint32_t>(rows);
    cm->width = msg->width;
    cm->encoding = msg->encoding;
    cm->is_bigendian = msg->is_bigendian;
    cm->step = msg->step;
    const uint8_t * src = msg->data.data() + static_cast<size_t>(r0) * step;
    cm->data.assign(src, src + rows * step);  // one memcpy of the contiguous row block

    auto ci = std::make_unique<CameraInfo>();
    ci->header = info->header;
    ci->width = info->width;
    ci->height = static_cast<uint32_t>(rows);
    ci->distortion_model = info->distortion_model;
    ci->d = info->d;
    ci->k = info->k;
    ci->r = info->r;
    ci->k[5] = K[5] - static_cast<double>(r0);
    ci->p = info->p;
    ci->p[6] = info->p[6] - static_cast<double>(r0);
    ci->roi.y_offset = static_cast<uint32_t>(r0);
    ci->roi.height = static_cast<uint32_t>(rows);
    ci->roi.width = info->width;

    cpub_img_[i]->publish(std::move(cm));
    cpub_info_[i]->publish(std::move(ci));
  }

  double baseline_{0.20};
  std::string in_ns_;
  std::string suffix_;
  int64_t crop_top_{340};
  int64_t crop_bottom_{624};
  int64_t pair_tol_ns_{8000000};
  double min_period_{0.1};
  double last_pub_{-1e9};
  std::map<int64_t, PairEntry> pairs_;  // original wall stamp (ns) -> entry
  std::array<std::array<double, 9>, 2> k_{};
  std::array<bool, 2> have_k_{false, false};

  std::array<rclcpp::Publisher<Image>::SharedPtr, 2> pub_img_;
  std::array<rclcpp::Publisher<CameraInfo>::SharedPtr, 2> pub_info_;
  std::array<rclcpp::Publisher<Image>::SharedPtr, 2> cpub_img_;
  std::array<rclcpp::Publisher<CameraInfo>::SharedPtr, 2> cpub_info_;
  std::array<rclcpp::Subscription<Image>::SharedPtr, 2> sub_img_;
  std::array<rclcpp::Subscription<CameraInfo>::SharedPtr, 2> sub_info_;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StampRelay>());
  rclcpp::shutdown();
  return 0;
}
