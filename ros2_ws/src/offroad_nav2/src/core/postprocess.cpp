#include "offroad_nav2/postprocess.hpp"

#include <algorithm>
#include <numeric>
#include <random>

namespace offroad {

namespace {
struct Lifted { Vec3 pc; bool ok = false; };

Lifted lift(double u, double v, const DepthView& depth, const Mat3& K, const PostParams& p) {
  Lifted r;
  const double z = median_depth(depth, u, v, p.win);
  if (std::isfinite(z) && z >= p.min_depth && z <= p.max_depth) {
    r.pc = backproject(u, v, z, K);
    r.ok = true;
  }
  return r;
}

inline void push(std::vector<float>& c, const Vec3& p) {
  c.push_back(float(p.x())); c.push_back(float(p.y())); c.push_back(float(p.z()));
}
}  // namespace

FrameClouds build_clouds(const uint8_t* seg, const uint8_t* contour, const uint8_t* objseg, const DepthView& depth, const Mat3& K,
                         const Mat4& T, const Vec3& up_ground, const PostParams& p, unsigned rng_seed) {
  FrameClouds out;
  const int nx = (MODEL_W + p.stride - 1) / p.stride, ny = (MODEL_H + p.stride - 1) / p.stride;
  const size_t N = size_t(nx) * ny;
  std::vector<Vec3, Eigen::aligned_allocator<Vec3>> pc(N, Vec3::Constant(kNaN));   // optical frame
  std::vector<double> h(N, kNaN);
  std::vector<char> ok(N, 0);
  Points pg(N);

  // lift depth only below the horizon (nothing above it is projected anyway): about half the work
  const auto hor = horizon_row(K, T);
  const Mat3 R = T.topLeftCorner<3, 3>();
  const Vec3 t = T.topRightCorner<3, 1>();
  for (int gy = 0; gy < ny; ++gy) {
    for (int gx = 0; gx < nx; ++gx) {
      double u, v;
      model_to_image_coords(gx * p.stride, gy * p.stride, depth.width, depth.height, p.crop, p.crop_top, p.crop_bottom, u, v);
      if (hor && v < *hor + p.horizon_margin_px) continue;
      const Lifted l = lift(u, v, depth, K, p);
      if (!l.ok) continue;
      const size_t i = size_t(gy) * nx + gx;
      pc[i] = l.pc;
      pg[i] = R * l.pc + t;
      ok[i] = 1;
    }
  }
  if (p.has_ego_box) {   // hood / bumper / mirrors seen by the camera are not obstacles
    const auto& b = p.ego_box;
    for (size_t i = 0; i < N; ++i)
      if (ok[i] && pg[i].x() > b[0] && pg[i].x() < b[1] && pg[i].y() > b[2] && pg[i].y() < b[3]) ok[i] = 0;
  }

  // --- ground plane from drivable pixels
  auto segAt = [&](int gy, int gx) { return seg[size_t(gy * p.stride) * MODEL_W + gx * p.stride]; };
  Vec3 n = up_ground.normalized();
  double d = 0.0;
  if (seg) {
    Points drv;
    for (int gy = 0; gy < ny; ++gy)
      for (int gx = 0; gx < nx; ++gx) {
        const size_t i = size_t(gy) * nx + gx;
        if (ok[i] && segAt(gy, gx) == 1 && pc[i].z() < 20.0) drv.push_back(pg[i]);
      }
    if (int(drv.size()) > p.ransac_max_pts) {   // random subsample without replacement
      std::mt19937 rng(rng_seed);
      for (int i = 0; i < p.ransac_max_pts; ++i) {
        std::uniform_int_distribution<size_t> pick(i, drv.size() - 1);
        std::swap(drv[i], drv[pick(rng)]);
      }
      drv.resize(p.ransac_max_pts);
    }
    const auto fit = fit_ground_plane(drv, up_ground, p.ransac_iters, p.ransac_thresh, p.max_tilt_deg, rng_seed);
    if (fit && std::abs(fit->d) < 0.6) {
      out.plane_fitted = true;
      n = fit->n;
      d = fit->d;
    }
  }
  out.plane_n = n;
  out.plane_d = d;
  size_t n_ok = 0;
  for (size_t i = 0; i < N; ++i) if (ok[i]) { h[i] = pg[i].dot(n) + d; ++n_ok; }
  out.valid_depth_frac = double(n_ok) / double(N);

  // --- geometric obstacles: anything standing above the ground, from depth alone
  auto& geo = out.clouds["geometric"];
  for (size_t i = 0; i < N; ++i)
    if (ok[i] && h[i] > p.obstacle_h_min && h[i] < p.obstacle_h_max) push(geo, pc[i]);

  if (seg) {
    auto& nd = out.clouds["nondrivable"];
    auto& clear = out.clouds["clear"];
    for (int gy = 0; gy < ny; ++gy)
      for (int gx = 0; gx < nx; ++gx) {
        const size_t i = size_t(gy) * nx + gx;
        if (!ok[i]) continue;
        const uint8_t s = segAt(gy, gx);
        if (s == 0 && std::abs(h[i]) < p.ground_h) push(nd, pc[i]);
        if (s == 1 && h[i] < p.ground_h && ((gx + gy) % 2 == 0)) push(clear, pc[i]);   // checkerboard thinning
      }
  }

  // --- semantic classes
  if (objseg) {
    for (int c = 1; c <= 4; ++c) {
      std::vector<char> keep(N, 0);
      for (int gx = 0; gx < nx; ++gx) {
        int lowest = -1;
        bool tall = false;
        for (int gy = 0; gy < ny; ++gy) {
          const size_t i = size_t(gy) * nx + gx;
          if (!ok[i] || objseg[size_t(gy * p.stride) * MODEL_W + gx * p.stride] != c) continue;
          if (h[i] > p.class_h_min && h[i] < p.class_h_max) keep[i] = 1;
          if (h[i] > p.lowest_h_min && h[i] < p.class_h_max) lowest = gy;       // lowest valid pixel of the column
          if (h[i] > p.tall_h) tall = true;                                      // a real standing object in this column
        }
        // ground contact only for columns that hold a standing object (paint marks classed "man-made" are flat)
        if (lowest >= 0 && tall) keep[size_t(lowest) * nx + gx] = 1;
      }
      auto& cl = out.clouds[class_name(c)];
      for (size_t i = 0; i < N; ++i) if (keep[i]) push(cl, pc[i]);
    }
  }

  // --- contour
  if (contour) {
    const ContourEndpoints e = contour_endpoints(contour);
    auto& boundary = out.clouds["contour_boundary"];
    auto& clear_end = out.clouds["contour_clear_end"];
    for (int i = 0; i < CONTOUR_RAYS; ++i) {
      double u, v;
      model_to_image_coords(e.col[i], e.row[i], depth.width, depth.height, p.crop, p.crop_top, p.crop_bottom, u, v);
      const Lifted l = lift(u, v, depth, K, p);
      if (!l.ok || above_horizon(v, K, T, p.horizon_margin_px)) continue;
      push(e.clear_only[i] ? clear_end : boundary, l.pc);
    }
    if (out.clouds.count("clear")) { auto& c = out.clouds["clear"]; c.insert(c.end(), clear_end.begin(), clear_end.end()); }
    if (p.contour_marks_nondrivable && out.clouds.count("nondrivable")) {
      auto& c = out.clouds["nondrivable"]; c.insert(c.end(), boundary.begin(), boundary.end());
    }
  }
  return out;
}

}  // namespace offroad
