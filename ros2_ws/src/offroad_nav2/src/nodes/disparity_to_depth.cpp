// C++ port of offroad_perception/disparity_to_depth.py.
//
// stereo_msgs/DisparityImage on /stereo/disparity -> /stereo/depth (32FC1, metres, left optical frame, NaN = invalid).
// The disparity image may cover only the matcher's crop; it is padded back to full_height starting at crop_top.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "stereo_msgs/msg/disparity_image.hpp"

namespace {

using sensor_msgs::msg::Image;
using stereo_msgs::msg::DisparityImage;

class DisparityToDepth : public rclcpp::Node
{
public:
  DisparityToDepth()
  : Node("disparity_to_depth")
  {
    min_depth_ = static_cast<float>(declare_parameter<double>("min_depth", 1.0));
    max_depth_ = static_cast<float>(declare_parameter<double>("max_depth", 60.0));
    crop_top_ = declare_parameter<int64_t>("crop_top", 340);
    full_h_ = declare_parameter<int64_t>("full_height", 720);

    const auto qos = rclcpp::QoS(2).reliable();
    pub_ = create_publisher<Image>("/stereo/depth", qos);
    sub_ = create_subscription<DisparityImage>(
      "/stereo/disparity", qos,
      [this](DisparityImage::ConstSharedPtr msg) {on_disparity(*msg);});
  }

private:
  void on_disparity(const DisparityImage & msg)
  {
    const Image & im = msg.image;
    const int64_t w = im.width;
    const int64_t h = im.height;
    const int64_t in_stride = im.step / 4;  // floats per input row
    if (im.step % 4 != 0 || in_stride < w || im.data.size() < static_cast<size_t>(h) * im.step) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "malformed disparity image, dropped");
      return;
    }

    // pad back to the full frame when the disparity covers only the crop
    const bool pad = full_h_ > h;
    const int64_t row0 = pad ? crop_top_ : 0;
    const int64_t out_rows = pad ? full_h_ : h;
    if (row0 < 0 || row0 > out_rows) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "crop_top outside the frame, dropped");
      return;
    }
    const int64_t n_copy = std::min(h, out_rows - row0);

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const float ft = static_cast<float>(msg.f * msg.t);  // numpy: float32 array divided by float(f*t)
    const float dmin = std::max(msg.min_disparity, 0.0f);
    const float * disp = reinterpret_cast<const float *>(im.data.data());

    auto out = std::make_unique<Image>();
    out->header = msg.header;
    out->height = static_cast<uint32_t>(out_rows);
    out->width = static_cast<uint32_t>(w);
    out->encoding = "32FC1";
    out->is_bigendian = 0;
    out->step = static_cast<uint32_t>(w * 4);
    out->data.resize(static_cast<size_t>(out_rows * w) * sizeof(float));
    float * dst = reinterpret_cast<float *>(out->data.data());  // vector storage is suitably aligned for float

    if (pad) {
      std::fill(dst, dst + row0 * w, nan);
      std::fill(dst + (row0 + n_copy) * w, dst + out_rows * w, nan);
    }
    const float zmin = min_depth_;
    const float zmax = max_depth_;
    for (int64_t r = 0; r < n_copy; ++r) {
      const float * src = disp + r * in_stride;
      float * o = dst + (row0 + r) * w;
      for (int64_t x = 0; x < w; ++x) {
        const float d = src[x];
        const float z = ft / d;
        // NaN and +-inf fail the range checks, so this is the same mask as isfinite(z) & range checks
        const bool ok = (d > dmin) & (z >= zmin) & (z <= zmax) & (z < inf);
        o[x] = ok ? z : nan;
      }
    }

    pub_->publish(std::move(out));
  }

  float min_depth_{1.0f};
  float max_depth_{60.0f};
  int64_t crop_top_{340};
  int64_t full_h_{720};
  rclcpp::Publisher<Image>::SharedPtr pub_;
  rclcpp::Subscription<DisparityImage>::SharedPtr sub_;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DisparityToDepth>());
  rclcpp::shutdown();
  return 0;
}
