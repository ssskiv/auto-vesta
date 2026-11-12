// Camera geometry (pure C++/Eigen, no ROS). Port of offroad_perception/geometry.py.
// Frames: optical = x right, y down, z forward; body (base_link / base_footprint) = x forward, y left, z up.
// Transforms are 4x4 and map child-frame points into the parent frame (T_parent_child).
#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

namespace offroad {

constexpr int MODEL_W = 640, MODEL_H = 320;
constexpr int CONTOUR_RAYS = 37, CONTOUR_BINS = 46, BIN_PX = 10;
constexpr int NO_HIT_BIN = CONTOUR_BINS - 1;   // 45: ray found nothing; never an obstacle
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

using Mat3 = Eigen::Matrix3d;
using Mat4 = Eigen::Matrix4d;
using Vec3 = Eigen::Vector3d;
using Points = std::vector<Vec3, Eigen::aligned_allocator<Vec3>>;

Mat3 rpy_to_R(double roll, double pitch, double yaw);
Mat4 make_T(const Vec3& xyz, double roll = 0, double pitch = 0, double yaw = 0);
// URDF optical-frame rotation for an x-forward body link: rpy = (-pi/2, 0, -pi/2)
Mat4 body_to_optical_T(const Vec3& xyz, double pitch_down = 0.0);
inline Vec3 transform(const Mat4& T, const Vec3& p) { return T.topLeftCorner<3, 3>() * p + T.topRightCorner<3, 1>(); }

// K of the original w x h image -> K of the 640x320 model input
Mat3 scale_intrinsics(const Mat3& K, int w, int h, bool crop = false, int crop_top = 0, int crop_bottom = 0);
// 640x320 model pixel coords -> original image pixel coords (plain scaling, exact inverse of scale_intrinsics)
void model_to_image_coords(double x, double y, int w, int h, bool crop, int crop_top, int crop_bottom, double& u, double& v);

// Depth image view: float metres, NaN / <= 0 invalid, row-major with `stride` floats per row.
struct DepthView {
  const float* data = nullptr;
  int width = 0, height = 0, stride = 0;
};
// Median of the valid (finite, > 0) depths in a win x win window around (u, v) (rounded half to even). NaN if none valid.
double median_depth(const DepthView& d, double u, double v, int win = 5);

inline Vec3 backproject(double u, double v, double z, const Mat3& K) {
  return {(u - K(0, 2)) * z / K(0, 0), (v - K(1, 2)) * z / K(1, 1), z};
}
inline void project(const Vec3& p, const Mat3& K, double& u, double& v) {
  u = K(0, 0) * p.x() / p.z() + K(0, 2);
  v = K(1, 1) * p.y() / p.z() + K(1, 2);
}

// Image row of the horizon at u = cx (rays with zero upward component in the ground frame). nullopt: no horizon in the image.
std::optional<double> horizon_row(const Mat3& K, const Mat4& T_ground_cam);
bool above_horizon(double v, const Mat3& K, const Mat4& T_ground_cam, double margin_px = 2.0);
// Intersect the ray through pixel (u, v) with the plane z = ground_z of the ground frame. nullopt: no hit in front of the camera.
std::optional<Vec3> pixel_to_ground(double u, double v, const Mat3& K, const Mat4& T_ground_cam, double ground_z = 0.0);

struct Plane {
  Vec3 n;           // unit normal, same side as up_prior
  double d = 0;     // n.p + d = 0
  std::vector<char> inliers;
};
// RANSAC plane on 3D points; candidate planes tilted more than max_tilt_deg from up_prior are rejected.
std::optional<Plane> fit_ground_plane(const Points& pts, const Vec3& up_prior, int iters = 200, double thresh = 0.08,
                                      double max_tilt_deg = 25.0, unsigned seed = 0, int min_pts = 30);

// Contour endpoints in the 640x320 image: rays go left->right from (row 319, col 320), end = start + 10 px * bin.
// clear_only is set for bin 45 and for rays that leave the image (those may clear space but never mark an obstacle).
struct ContourEndpoints {
  double row[CONTOUR_RAYS], col[CONTOUR_RAYS];
  bool clear_only[CONTOUR_RAYS];
};
ContourEndpoints contour_endpoints(const uint8_t bins[CONTOUR_RAYS]);

}  // namespace offroad
