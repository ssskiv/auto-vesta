// Network outputs + depth -> point clouds per class (pure C++, no ROS). Port of offroad_perception/postprocess.py.
// All returned point sets are in the camera OPTICAL frame (flat x,y,z float triples), ready to be stamped with the image time.
// Heights are measured against a ground plane fitted per frame in the footprint frame.
#pragma once
#include <array>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "offroad_nav2/geometry.hpp"

namespace offroad {

// objseg classes: 0 background, 1 man-made, 2 vulnerable living (vru), 3 vehicle, 4 natural obstacle
inline const char* class_name(int c) {
  switch (c) { case 1: return "manmade"; case 2: return "vru"; case 3: return "vehicle"; case 4: return "natural"; default: return ""; }
}

struct PostParams {
  int stride = 4;                 // model-pixel grid stride for dense clouds
  int win = 5;                    // median-depth window
  double min_depth = 1.5;
  double max_depth = 20.0;        // stereo error ~ z^2: trust depth only to ~20 m
  double obstacle_h_min = 0.15, obstacle_h_max = 2.2;
  double ground_h = 0.15;         // |h| below this = flat ground
  double horizon_margin_px = 3.0;
  int ransac_iters = 150;
  double ransac_thresh = 0.08;
  int ransac_max_pts = 1500;
  double max_tilt_deg = 20.0;
  double class_h_min = 0.05, class_h_max = 2.5;
  double tall_h = 0.3;            // a column needs a point this high for its ground-contact point to count
  double lowest_h_min = -0.3;
  bool contour_marks_nondrivable = true;
  bool has_ego_box = false;       // points inside (xmin, xmax, ymin, ymax) in the ground frame are the car itself
  std::array<double, 4> ego_box{0, 0, 0, 0};
  bool crop = false;              // preprocess mode 'crop' (else 'resize')
  int crop_top = 0, crop_bottom = 0;
};

struct FrameClouds {
  std::map<std::string, std::vector<float>> clouds;   // name -> flat x,y,z (optical frame)
  bool plane_fitted = false;
  Vec3 plane_n = Vec3::UnitZ();
  double plane_d = 0;
  double valid_depth_frac = 0;
};

// seg / objseg: 320x640 class maps or nullptr; contour: 37 bins or nullptr; depth: full-resolution metres.
// K: full-resolution intrinsics; T_ground_cam: footprint <- optical; up_ground: expected up vector in the ground frame.
FrameClouds build_clouds(const uint8_t* seg, const uint8_t* contour, const uint8_t* objseg, const DepthView& depth, const Mat3& K,
                         const Mat4& T_ground_cam, const Vec3& up_ground, const PostParams& p, unsigned rng_seed = 0);

}  // namespace offroad
