#include "offroad_nav2/geometry.hpp"

#include <algorithm>
#include <random>

namespace offroad {

Mat3 rpy_to_R(double roll, double pitch, double yaw) {
  return (Eigen::AngleAxisd(yaw, Vec3::UnitZ()) * Eigen::AngleAxisd(pitch, Vec3::UnitY()) *
          Eigen::AngleAxisd(roll, Vec3::UnitX())).toRotationMatrix();
}

Mat4 make_T(const Vec3& xyz, double roll, double pitch, double yaw) {
  Mat4 T = Mat4::Identity();
  T.topLeftCorner<3, 3>() = rpy_to_R(roll, pitch, yaw);
  T.topRightCorner<3, 1>() = xyz;
  return T;
}

Mat4 body_to_optical_T(const Vec3& xyz, double pitch_down) {
  return make_T(xyz, 0.0, pitch_down, 0.0) * make_T(Vec3::Zero(), -M_PI / 2, 0.0, -M_PI / 2);
}

Mat3 scale_intrinsics(const Mat3& K0, int w, int h, bool crop, int crop_top, int crop_bottom) {
  Mat3 K = K0;
  if (crop) {
    K(1, 2) -= crop_top;
    h = h - crop_top - crop_bottom;
  }
  const double sx = double(MODEL_W) / w, sy = double(MODEL_H) / h;
  K.row(0) *= sx;
  K.row(1) *= sy;
  return K;
}

void model_to_image_coords(double x, double y, int w, int h, bool crop, int crop_top, int crop_bottom, double& u, double& v) {
  const int hh = crop ? h - crop_top - crop_bottom : h;
  u = x * (double(w) / MODEL_W);
  v = y * (double(hh) / MODEL_H) + (crop ? crop_top : 0);
}

double median_depth(const DepthView& d, double u, double v, int win) {
  const int r = win / 2;
  const int ui = int(std::nearbyint(u)), vi = int(std::nearbyint(v));
  if (ui < 0 || ui >= d.width || vi < 0 || vi >= d.height) return kNaN;
  float buf[64];
  int n = 0;
  for (int dy = -r; dy <= r; ++dy) {
    const int y = std::clamp(vi + dy, 0, d.height - 1);
    const float* row = d.data + size_t(y) * d.stride;
    for (int dx = -r; dx <= r; ++dx) {
      const float z = row[std::clamp(ui + dx, 0, d.width - 1)];
      if (std::isfinite(z) && z > 0.f && n < 64) buf[n++] = z;
    }
  }
  if (n == 0) return kNaN;
  const int m = n / 2;
  std::nth_element(buf, buf + m, buf + n);
  if (n & 1) return buf[m];
  const float hi = buf[m];
  const float lo = *std::max_element(buf, buf + m);
  return 0.5 * (double(lo) + double(hi));
}

std::optional<double> horizon_row(const Mat3& K, const Mat4& T) {
  const double r21 = T(2, 1), r22 = T(2, 2);
  if (std::abs(r21) < 1e-9) return std::nullopt;
  return K(1, 2) - K(1, 1) * r22 / r21;
}

bool above_horizon(double v, const Mat3& K, const Mat4& T, double margin_px) {
  const auto h = horizon_row(K, T);
  return h ? v < *h + margin_px : false;
}

std::optional<Vec3> pixel_to_ground(double u, double v, const Mat3& K, const Mat4& T, double ground_z) {
  const Vec3 d = T.topLeftCorner<3, 3>() * backproject(u, v, 1.0, K);
  const Vec3 o = T.topRightCorner<3, 1>();
  if (std::abs(d.z()) < 1e-12) return std::nullopt;
  const double s = (ground_z - o.z()) / d.z();
  if (!std::isfinite(s) || s <= 0) return std::nullopt;
  return o + d * s;
}

std::optional<Plane> fit_ground_plane(const Points& pts, const Vec3& up_prior, int iters, double thresh, double max_tilt_deg,
                                      unsigned seed, int min_pts) {
  const int N = int(pts.size());
  if (N < min_pts) return std::nullopt;
  const Vec3 up = up_prior.normalized();
  const double cos_max = std::cos(max_tilt_deg * M_PI / 180.0);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> pick(0, N - 1);
  Vec3 best_n = Vec3::Zero();
  double best_d = 0;
  int best_count = -1;
  for (int it = 0; it < iters; ++it) {
    const int i = pick(rng), j = pick(rng), k = pick(rng);
    if (i == j || j == k || i == k) continue;
    Vec3 n = (pts[j] - pts[i]).cross(pts[k] - pts[i]);
    const double nn = n.norm();
    if (nn < 1e-9) continue;
    n /= nn;
    if (n.dot(up) < 0) n = -n;
    if (n.dot(up) < cos_max) continue;
    const double d = -n.dot(pts[i]);
    int c = 0;
    for (const auto& p : pts) c += std::abs(n.dot(p) + d) < thresh;
    if (c > best_count) { best_count = c; best_n = n; best_d = d; }
  }
  if (best_count < min_pts) return std::nullopt;
  // least-squares refit on the inliers: normal = eigenvector of the smallest eigenvalue of the covariance
  Vec3 mean = Vec3::Zero();
  int m = 0;
  for (const auto& p : pts) if (std::abs(best_n.dot(p) + best_d) < thresh) { mean += p; ++m; }
  mean /= m;
  Mat3 C = Mat3::Zero();
  for (const auto& p : pts) if (std::abs(best_n.dot(p) + best_d) < thresh) C += (p - mean) * (p - mean).transpose();
  Eigen::SelfAdjointEigenSolver<Mat3> es(C);
  Vec3 n2 = es.eigenvectors().col(0);
  if (n2.dot(up) < 0) n2 = -n2;
  Plane out;
  out.n = n2;
  out.d = -n2.dot(mean);
  out.inliers.resize(N);
  for (int i = 0; i < N; ++i) out.inliers[i] = std::abs(n2.dot(pts[i]) + out.d) < thresh;
  return out;
}

ContourEndpoints contour_endpoints(const uint8_t bins[CONTOUR_RAYS]) {
  ContourEndpoints e;
  for (int i = 0; i < CONTOUR_RAYS; ++i) {
    const double th = M_PI + (0.0 - M_PI) * double(i) / (CONTOUR_RAYS - 1);   // linspace(pi, 0, 37)
    const double r = double(BIN_PX) * bins[i];
    const double row = 319 - r * std::sin(th), col = 320 + r * std::cos(th);
    const bool outside = row < 0 || row > MODEL_H - 1 || col < 0 || col > MODEL_W - 1;
    e.clear_only[i] = bins[i] >= NO_HIT_BIN || outside;
    e.row[i] = std::clamp(row, 0.0, double(MODEL_H - 1));
    e.col[i] = std::clamp(col, 0.0, double(MODEL_W - 1));
  }
  return e;
}

}  // namespace offroad
