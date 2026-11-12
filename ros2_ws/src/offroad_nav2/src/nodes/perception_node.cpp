// perception_node (C++): one inference engine (three separate networks on one preprocessed frame, ONNX Runtime/CUDA) + stereo depth
// -> point clouds per class. Depth-provider contract: /stereo/depth (32FC1 metres, left optical frame, same stamp as the image) +
// /stereo/left/camera_info. Every output carries the IMAGE stamp and the optical frame_id. One worker thread owns the GPU; the ROS
// callbacks only overwrite a one-slot "latest frame" buffer. C++ port of offroad_perception/perception_node.py.
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <opencv2/imgproc.hpp>
#include <thread>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float32.hpp>
#include <tf2/time.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <vision_msgs/msg/detection3_d_array.hpp>

#include "offroad_nav2/engine.hpp"
#include "offroad_nav2/geometry.hpp"
#include "offroad_nav2/postprocess.hpp"
#include "offroad_nav2/preprocess.hpp"
#include "offroad_nav2/tracking.hpp"

using namespace offroad;
using Clock = std::chrono::steady_clock;
using sensor_msgs::msg::Image;

namespace {
const char* const kMarking[] = {"geometric", "nondrivable", "manmade", "vru", "vehicle", "natural"};   // temporally filtered
const char* const kClouds[] = {"geometric", "nondrivable", "clear", "vru", "vehicle", "manmade", "natural", "contour_boundary",
                               "contour_clear_end"};
bool is_marking(const std::string& n) { return std::find(std::begin(kMarking), std::end(kMarking), n) != std::end(kMarking); }

double pct(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, size_t(q * v.size()))];
}
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
}  // namespace

class PerceptionNode : public rclcpp::Node {
 public:
  PerceptionNode() : Node("perception_node"), tf_buffer_(get_clock()), tf_listener_(tf_buffer_) {
    ecfg_.onnx_dir = declare_parameter<std::string>("onnx_dir", "/models/onnx");
    models_enabled_ = declare_parameter<bool>("models_enabled", true);
    ecfg_.parallel = declare_parameter<bool>("parallel_nets", false);
    pp_.stride = int(declare_parameter<int>("stride", 4));
    pp_.max_depth = declare_parameter<double>("max_depth", 20.0);
    const std::string mode = declare_parameter<std::string>("preprocess_mode", "resize");   // TODO(user) decision 7.8
    pp_.crop = mode == "crop";
    pp_.crop_top = int(declare_parameter<int>("crop_top", 40));
    pp_.crop_bottom = int(declare_parameter<int>("crop_bottom", 40));
    pp_.contour_marks_nondrivable = declare_parameter<bool>("contour_marks_nondrivable", true);
    // footprint x -0.88..3.68, y +-0.865 (vehicle.yaml) + 0.2 m behind/side, +0.5 m ahead (hood / bumper)
    const auto eb = declare_parameter<std::vector<double>>("ego_box", {-1.08, 4.18, -1.065, 1.065});
    if (eb.size() == 4) { pp_.has_ego_box = true; for (int i = 0; i < 4; ++i) pp_.ego_box[i] = eb[i]; }
    footprint_ = declare_parameter<std::string>("footprint_frame", "base_footprint");
    odom_ = declare_parameter<std::string>("odom_frame", "odom");
    temporal_ = declare_parameter<bool>("temporal_filter", true);
    tf_timeout_ = declare_parameter<double>("tf_timeout_s", 0.05);
    const std::string image_topic = declare_parameter<std::string>("image_topic", "/stereo/left/image_rect");
    for (const char* n : kMarking) filters_.emplace(n, TemporalFilter(0.2, 3, 2));
    trackers_.emplace("vru", CentroidTracker());
    trackers_.emplace("vehicle", CentroidTracker());

    for (const char* n : kClouds) pub_[n] = create_publisher<sensor_msgs::msg::PointCloud2>(std::string("/perception/points/") + n, 2);
    pub_obj_ = create_publisher<vision_msgs::msg::Detection3DArray>("/perception/objects", 2);
    pub_lat_ = create_publisher<std_msgs::msg::Float32>("/perception/latency_ms", 2);
    pub_vis_ = create_publisher<Image>("/perception/seg_vis", 1);
    const auto reliable = rclcpp::QoS(2).reliable();
    sub_img_ = create_subscription<Image>(image_topic, reliable, [this](Image::ConstSharedPtr m) { on_image(std::move(m)); });
    sub_depth_ = create_subscription<Image>("/stereo/depth", reliable, [this](Image::ConstSharedPtr m) { on_depth(std::move(m)); });
    sub_info_ = create_subscription<sensor_msgs::msg::CameraInfo>("/stereo/left/camera_info", reliable,
        [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr m) {
          Mat3 K;
          for (int i = 0; i < 9; ++i) K(i / 3, i % 3) = m->k[i];
          std::lock_guard<std::mutex> l(mtx_);
          K_ = K; has_K_ = true;
        });
    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>("/imu/data", 5, [this](sensor_msgs::msg::Imu::ConstSharedPtr m) {
      const Eigen::Quaterniond q(m->orientation.w, m->orientation.x, m->orientation.y, m->orientation.z);
      std::lock_guard<std::mutex> l(mtx_);
      up_ = q.toRotationMatrix().transpose() * Vec3::UnitZ();   // world up in base_link axes
    });
    report_timer_ = create_wall_timer(std::chrono::seconds(10), [this] { report(); });
    worker_ = std::thread([this] { worker(); });
  }

  ~PerceptionNode() override {
    stop_ = true;
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

 private:
  using StampKey = std::pair<int32_t, uint32_t>;

  void on_image(Image::ConstSharedPtr m) {
    std::lock_guard<std::mutex> l(mtx_);
    images_[{m->header.stamp.sec, m->header.stamp.nanosec}] = std::move(m);
    while (images_.size() > 6) images_.erase(images_.begin());
  }

  void on_depth(Image::ConstSharedPtr dm) {
    {
      std::lock_guard<std::mutex> l(mtx_);
      auto it = images_.find({dm->header.stamp.sec, dm->header.stamp.nanosec});
      if (it == images_.end()) return;
      slot_img_ = it->second;
      slot_depth_ = std::move(dm);   // overwrite: latest frame wins
    }
    cv_.notify_one();
  }

  void worker() {
    try {
      if (models_enabled_) {
        engine_ = std::make_unique<Engine>(ecfg_);
        RCLCPP_INFO(get_logger(), "engine ready (%s, onnx dir %s, parallel=%d)", engine_->provider().c_str(), ecfg_.onnx_dir.c_str(),
                    int(ecfg_.parallel));
      }
    } catch (const std::exception& e) {
      RCLCPP_FATAL(get_logger(), "engine failed to start: %s", e.what());
      rclcpp::shutdown();
      return;
    }
    std::vector<uint8_t> chw(size_t(3) * MODEL_H * MODEL_W);
    while (rclcpp::ok() && !stop_) {
      Image::ConstSharedPtr im, dm;
      {
        std::unique_lock<std::mutex> l(mtx_);
        cv_.wait_for(l, std::chrono::milliseconds(500), [this] { return slot_depth_ != nullptr || stop_; });
        if (!slot_depth_) continue;
        im = std::move(slot_img_); dm = std::move(slot_depth_);
        slot_img_.reset(); slot_depth_.reset();
      }
      try {
        process(*im, *dm, chw);
      } catch (const std::exception& e) {   // keep the node alive
        RCLCPP_ERROR(get_logger(), "frame failed: %s", e.what());
      }
    }
  }

  static sensor_msgs::msg::PointCloud2 make_cloud(const std::vector<float>& xyz, const builtin_interfaces::msg::Time& stamp,
                                                  const std::string& frame) {
    sensor_msgs::msg::PointCloud2 m;
    m.header.stamp = stamp;
    m.header.frame_id = frame;
    m.height = 1;
    m.width = uint32_t(xyz.size() / 3);
    m.fields.resize(3);
    const char* names[] = {"x", "y", "z"};
    for (int i = 0; i < 3; ++i) { m.fields[i].name = names[i]; m.fields[i].offset = 4 * i; m.fields[i].datatype = sensor_msgs::msg::PointField::FLOAT32; m.fields[i].count = 1; }
    m.is_bigendian = false; m.point_step = 12; m.is_dense = true; m.row_step = 12 * m.width;
    m.data.resize(xyz.size() * sizeof(float));
    if (!xyz.empty()) std::memcpy(m.data.data(), xyz.data(), m.data.size());
    return m;
  }

  void process(const Image& im, const Image& dm, std::vector<uint8_t>& chw) {
    Mat3 K; Vec3 up;
    {
      std::lock_guard<std::mutex> l(mtx_);
      if (!has_K_) return;
      K = K_; up = up_;
    }
    const auto t_start = Clock::now();
    wait_ms_.push_back((now() - rclcpp::Time(dm.header.stamp, get_clock()->get_clock_type())).seconds() * 1e3);
    const std::string frame = im.header.frame_id;
    if (!have_T_fp_) {
      try {
        T_fp_cam_ = tf2::transformToEigen(tf_buffer_.lookupTransform(footprint_, frame, tf2::TimePointZero)).matrix();
        have_T_fp_ = true;
      } catch (const std::exception&) { return; }
    }
    const int ch = im.encoding.find("a8") != std::string::npos ? 4 : 3;
    const bool bgr = im.encoding.rfind("bgr", 0) == 0;
    DepthView depth{reinterpret_cast<const float*>(dm.data.data()), int(dm.width), int(dm.height), int(dm.step / 4)};

    // --- networks
    const uint8_t *seg = nullptr, *contour = nullptr, *objseg = nullptr;
    EngineResult er;
    if (engine_) {
      preprocess_rgb_chw(im.data.data(), int(im.width), int(im.height), int(im.step), ch, bgr, pp_.crop, pp_.crop_top, pp_.crop_bottom,
                         chw.data());
      stage_["pre"].push_back(ms_since(t_start));
      engine_->infer(chw.data(), er);
      if (!er.seg.empty()) seg = er.seg.data();
      if (er.has_contour) contour = er.contour;
      if (!er.objseg.empty()) objseg = er.objseg.data();
      stage_["nets"].push_back(er.ms_total);
    }
    const auto t_clouds = Clock::now();
    FrameClouds fc = build_clouds(seg, contour, objseg, depth, K, T_fp_cam_, up, pp_);
    stage_["clouds"].push_back(ms_since(t_clouds));
    if (fc.plane_fitted) plane_log_.emplace_back(fc.plane_d, std::acos(std::clamp(fc.plane_n.z(), -1.0, 1.0)) * 180.0 / M_PI);

    // --- temporal filter + objects need odom<-optical at the image time
    const auto t_pub = Clock::now();
    const auto& stamp = im.header.stamp;
    bool have_odom = false;
    Mat4 T_odom;
    try {
      T_odom = tf2::transformToEigen(tf_buffer_.lookupTransform(odom_, frame, rclcpp::Time(stamp), tf2::durationFromSec(tf_timeout_))).matrix();
      have_odom = true;
    } catch (const std::exception&) {}
    std::map<std::string, std::vector<Vec2, Eigen::aligned_allocator<Vec2>>> clusters_xy;
    for (auto& [name, pts] : fc.clouds) {
      const size_t n = pts.size() / 3;
      const bool tracked = trackers_.count(name) > 0;
      if (n && have_odom && (tracked || (temporal_ && is_marking(name)))) {
        std::vector<Vec2, Eigen::aligned_allocator<Vec2>> xy(n);
        const Mat3 R = T_odom.topLeftCorner<3, 3>();
        const Vec3 t = T_odom.topRightCorner<3, 1>();
        for (size_t i = 0; i < n; ++i) {
          const Vec3 p = R * Vec3(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]) + t;
          xy[i] = Vec2(p.x(), p.y());
        }
        if (temporal_ && is_marking(name)) {
          const auto keep = filters_.at(name).update(xy);
          std::vector<float> kept;
          std::vector<Vec2, Eigen::aligned_allocator<Vec2>> kxy;
          for (size_t i = 0; i < n; ++i)
            if (keep[i]) { kept.insert(kept.end(), pts.begin() + 3 * i, pts.begin() + 3 * i + 3); kxy.push_back(xy[i]); }
          if (tracked) clusters_xy[name] = std::move(kxy);
          pts = std::move(kept);
        } else if (tracked) {
          clusters_xy[name] = std::move(xy);
        }
      }
      auto it = pub_.find(name);
      if (it != pub_.end()) it->second->publish(make_cloud(pts, stamp, frame));
    }
    publish_objects(clusters_xy, stamp);
    stage_["filter+publish"].push_back(ms_since(t_pub));

    const double lat = (now() - rclcpp::Time(stamp, get_clock()->get_clock_type())).seconds() * 1e3;
    lat_ms_.push_back(lat);
    std_msgs::msg::Float32 lm; lm.data = float(lat);
    pub_lat_->publish(lm);
    ++frames_;
    if (frames_ % 6 == 0 && seg && pub_vis_->get_subscription_count() > 0) publish_vis(chw.data(), seg, objseg, stamp);
    keep_last(wait_ms_); keep_last(lat_ms_);
    for (auto& [k, v] : stage_) keep_last(v);
    while (plane_log_.size() > 200) plane_log_.pop_front();
  }

  template <class C> static void keep_last(C& v) { while (v.size() > 200) v.pop_front(); }

  void publish_objects(const std::map<std::string, std::vector<Vec2, Eigen::aligned_allocator<Vec2>>>& clusters_xy,
                       const builtin_interfaces::msg::Time& stamp) {
    vision_msgs::msg::Detection3DArray arr;
    arr.header.stamp = stamp;
    arr.header.frame_id = odom_;
    for (const auto& [name, xy] : clusters_xy) {
      for (const auto& [tid, c] : trackers_.at(name).update(cluster_bev(xy))) {
        vision_msgs::msg::Detection3D d;
        d.header = arr.header;
        d.id = name + "-" + std::to_string(tid);
        vision_msgs::msg::ObjectHypothesisWithPose hyp;
        hyp.hypothesis.class_id = name;
        hyp.hypothesis.score = 1.0;
        d.results.push_back(hyp);
        d.bbox.center.position.x = c.center.x(); d.bbox.center.position.y = c.center.y(); d.bbox.center.position.z = 0.0;
        d.bbox.size.x = c.size.x(); d.bbox.size.y = c.size.y(); d.bbox.size.z = 1.0;
        arr.detections.push_back(d);
      }
    }
    pub_obj_->publish(arr);
  }

  void publish_vis(const uint8_t* chw, const uint8_t* seg, const uint8_t* objseg, const builtin_interfaces::msg::Time& stamp) {
    const size_t plane = size_t(MODEL_W) * MODEL_H;
    auto m = std::make_unique<Image>();
    m->header.stamp = stamp; m->header.frame_id = "seg_vis";
    m->height = MODEL_H; m->width = MODEL_W; m->encoding = "rgb8"; m->step = MODEL_W * 3;
    m->data.resize(plane * 3);
    const int cols[5][3] = {{0, 0, 0}, {255, 255, 0}, {255, 0, 0}, {0, 0, 255}, {255, 0, 255}};
    for (size_t i = 0; i < plane; ++i) {
      int rgb[3] = {chw[i], chw[plane + i], chw[2 * plane + i]};
      if (seg[i] == 1) { rgb[0] = int(0.55 * rgb[0]); rgb[1] = int(0.55 * rgb[1] + 0.45 * 255); rgb[2] = int(0.55 * rgb[2]); }
      if (objseg && objseg[i] >= 1 && objseg[i] <= 4)
        for (int c = 0; c < 3; ++c) rgb[c] = int(0.5 * rgb[c] + 0.5 * cols[objseg[i]][c]);
      for (int c = 0; c < 3; ++c) m->data[3 * i + c] = uint8_t(rgb[c]);
    }
    pub_vis_->publish(std::move(m));
  }

  void report() {
    if (lat_ms_.empty()) return;
    const std::vector<double> lat(lat_ms_.begin(), lat_ms_.end()), wait(wait_ms_.begin(), wait_ms_.end());
    std::string s = "image->cloud latency p50 " + fmt(pct(lat, 0.5)) + " ms p95 " + fmt(pct(lat, 0.95)) + " ms (" +
                    std::to_string(lat.size()) + " frames, " + std::to_string(frames_) + " total); depth-arrival->now p50 " +
                    fmt(pct(wait, 0.5)) + " ms; ";
    if (!plane_log_.empty()) {
      std::vector<double> d, tl;
      for (auto& p : plane_log_) { d.push_back(p.first); tl.push_back(p.second); }
      s += "ground plane in " + footprint_ + ": d median " + fmt(pct(d, 0.5), 3) + " m, tilt " + fmt(pct(tl, 0.5), 2) + " deg; ";
    }
    for (auto& [k, v] : stage_) s += k + " " + fmt(pct(std::vector<double>(v.begin(), v.end()), 0.5)) + " ";
    RCLCPP_INFO(get_logger(), "%s(stage p50 ms)", s.c_str());
  }
  static std::string fmt(double v, int prec = 0) { char b[32]; std::snprintf(b, sizeof b, "%.*f", prec, v); return b; }

  // parameters / state
  EngineConfig ecfg_;
  PostParams pp_;
  bool models_enabled_ = true, temporal_ = true;
  std::string footprint_, odom_;
  double tf_timeout_ = 0.05;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  Mat3 K_ = Mat3::Identity();
  bool has_K_ = false, have_T_fp_ = false;
  Vec3 up_ = Vec3::UnitZ();
  Mat4 T_fp_cam_ = Mat4::Identity();
  std::map<StampKey, Image::ConstSharedPtr> images_;
  Image::ConstSharedPtr slot_img_, slot_depth_;
  std::mutex mtx_;
  std::condition_variable cv_;
  std::atomic<bool> stop_{false};
  std::thread worker_;
  std::unique_ptr<Engine> engine_;
  std::map<std::string, TemporalFilter> filters_;
  std::map<std::string, CentroidTracker> trackers_;
  std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> pub_;
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr pub_obj_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_lat_;
  rclcpp::Publisher<Image>::SharedPtr pub_vis_;
  rclcpp::Subscription<Image>::SharedPtr sub_img_, sub_depth_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_info_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::TimerBase::SharedPtr report_timer_;
  // statistics (worker thread only, except report() which only reads: guarded by being on the same logical sequence, copies taken)
  std::deque<double> lat_ms_, wait_ms_;
  std::map<std::string, std::deque<double>> stage_;
  std::deque<std::pair<double, double>> plane_log_;
  size_t frames_ = 0;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PerceptionNode>();
  rclcpp::executors::MultiThreadedExecutor ex(rclcpp::ExecutorOptions(), 3);
  ex.add_node(node);
  ex.spin();
  rclcpp::shutdown();
  return 0;
}
