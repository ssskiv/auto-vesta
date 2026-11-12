#include <gtest/gtest.h>

#include "offroad_nav2/geometry.hpp"
#include "offroad_nav2/postprocess.hpp"
#include "offroad_nav2/preprocess.hpp"
#include "offroad_nav2/tracking.hpp"

using namespace offroad;

namespace {
constexpr int W = 1280, H = 720;
Mat3 Kcam() { Mat3 K = Mat3::Identity(); K(0, 0) = K(1, 1) = 1171.5; K(0, 2) = 640; K(1, 2) = 360; return K; }

struct Scene {
  std::vector<float> depth = std::vector<float>(size_t(W) * H, std::nanf(""));
  std::vector<uint8_t> seg = std::vector<uint8_t>(size_t(MODEL_W) * MODEL_H, 0), obj = std::vector<uint8_t>(size_t(MODEL_W) * MODEL_H, 0);
};

// flat ground + a 0.9 m wide, 1.4 m tall box with its front face at x = 11 m (ground frame); camera 1.45 m up, level
Scene make_scene(const Mat4& T, const Mat3& K) {
  Scene s;
  const Mat3 R = T.topLeftCorner<3, 3>();
  const Vec3 o = T.topRightCorner<3, 1>();
  std::vector<char> box(size_t(W) * H, 0);
  for (int v = 0; v < H; ++v)
    for (int u = 0; u < W; ++u) {
      const Vec3 d = R * backproject(u, v, 1.0, K);
      double z = std::nan("");
      if (d.z() < -1e-9) { const double sg = (0 - o.z()) / d.z(); if (sg > 0 && sg < 60) z = sg; }
      if (d.x() > 1e-9) {
        const double sb = (11.0 - o.x()) / d.x(), yb = o.y() + sb * d.y(), zb = o.z() + sb * d.z();
        if (sb > 0 && std::abs(yb) < 0.45 && zb > 0 && zb < 1.4 && (std::isnan(z) || sb < z)) { z = sb; box[size_t(v) * W + u] = 1; }
      }
      s.depth[size_t(v) * W + u] = float(z);
    }
  for (int y = 0; y < MODEL_H; ++y)
    for (int x = 0; x < MODEL_W; ++x) {
      double u, v; model_to_image_coords(x, y, W, H, false, 0, 0, u, v);
      const size_t i = size_t(std::clamp(int(std::nearbyint(v)), 0, H - 1)) * W + std::clamp(int(std::nearbyint(u)), 0, W - 1);
      const bool has = std::isfinite(s.depth[i]);
      s.seg[size_t(y) * MODEL_W + x] = has && !box[i];
      s.obj[size_t(y) * MODEL_W + x] = box[i] ? 3 : 0;
    }
  return s;
}
Points ground_frame(const std::vector<float>& c, const Mat4& T) {
  Points p;
  for (size_t i = 0; i + 2 < c.size(); i += 3) p.push_back(transform(T, Vec3(c[i], c[i + 1], c[i + 2])));
  return p;
}
}  // namespace

TEST(Postprocess, ObstacleGroundAndHorizon) {
  const Mat3 K = Kcam(); const Mat4 T = body_to_optical_T(Vec3(1.55, 0.1, 1.45));
  Scene s = make_scene(T, K);
  uint8_t contour[CONTOUR_RAYS]; std::fill(contour, contour + CONTOUR_RAYS, 10);
  const DepthView dv{s.depth.data(), W, H, W};
  const FrameClouds fc = build_clouds(s.seg.data(), contour, s.obj.data(), dv, K, T, Vec3::UnitZ(), PostParams{});
  ASSERT_TRUE(fc.plane_fitted);
  EXPECT_LT(std::abs(fc.plane_d), 0.02);
  const Points geo = ground_frame(fc.clouds.at("geometric"), T);
  ASSERT_GT(geo.size(), 50u);
  int near_face = 0;
  for (const auto& p : geo) { near_face += std::abs(p.x() - 11.0) < 0.1; EXPECT_LT(std::abs(p.y()), 0.5); EXPECT_GT(p.z(), 0.1); EXPECT_LT(p.z(), 1.5); }
  EXPECT_GT(double(near_face) / geo.size(), 0.95);
  for (const auto& p : ground_frame(fc.clouds.at("clear"), T)) EXPECT_LT(std::abs(p.z()), 0.15);
  const Points veh = ground_frame(fc.clouds.at("vehicle"), T);
  ASSERT_GT(veh.size(), 20u);
  for (const auto& p : veh) EXPECT_LT(std::abs(p.x() - 11.0), 0.1);
  for (const auto& [name, c] : fc.clouds)   // nothing above the horizon
    for (size_t i = 0; i + 2 < c.size(); i += 3) {
      double u, v; project(Vec3(c[i], c[i + 1], c[i + 2]), K, u, v);
      EXPECT_FALSE(above_horizon(v, K, T, 0.0)) << name;
    }
}

TEST(Postprocess, ContourEdgeNeverObstacle) {
  const Mat3 K = Kcam(); const Mat4 T = body_to_optical_T(Vec3(1.55, 0.1, 1.45));
  Scene s = make_scene(T, K);
  uint8_t contour[CONTOUR_RAYS]; std::fill(contour, contour + CONTOUR_RAYS, 45);
  const FrameClouds fc = build_clouds(s.seg.data(), contour, s.obj.data(), DepthView{s.depth.data(), W, H, W}, K, T, Vec3::UnitZ(), PostParams{});
  EXPECT_EQ(fc.clouds.at("contour_boundary").size(), 0u);
  EXPECT_GE(fc.clouds.at("contour_clear_end").size(), 3u);
}

TEST(Postprocess, EgoBoxRemovesOwnBody) {
  const Mat3 K = Kcam(); const Mat4 T = body_to_optical_T(Vec3(1.55, 0.1, 1.45));
  Scene s = make_scene(T, K);
  const DepthView dv{s.depth.data(), W, H, W};
  PostParams p; p.has_ego_box = true; p.ego_box = {-1.08, 4.18, -1.065, 1.065};
  const Points g0 = ground_frame(build_clouds(s.seg.data(), nullptr, s.obj.data(), dv, K, T, Vec3::UnitZ(), PostParams{}).clouds.at("geometric"), T);
  const Points g1 = ground_frame(build_clouds(s.seg.data(), nullptr, s.obj.data(), dv, K, T, Vec3::UnitZ(), p).clouds.at("geometric"), T);
  ASSERT_GT(g1.size(), 0u);
  EXPECT_LE(g1.size(), g0.size());
  for (const auto& q : g1) EXPECT_GT(q.x(), 4.18);
}

TEST(Tracking, TemporalFilterClusterAndTracker) {
  TemporalFilter tf(0.2, 3, 2);
  std::vector<Vec2, Eigen::aligned_allocator<Vec2>> a{{1.0, 1.0}, {5.0, 5.0}};
  auto k = tf.update(a);
  EXPECT_FALSE(k[0] || k[1]);                       // first frame: nothing confirmed
  k = tf.update({{1.05, 1.02}, {9.0, 9.0}});
  EXPECT_TRUE(k[0]); EXPECT_FALSE(k[1]);            // repeated cell confirmed, new cell not
  std::vector<Vec2, Eigen::aligned_allocator<Vec2>> pts;
  for (int i = 0; i < 30; ++i) { pts.emplace_back(10 + 0.02 * i, 2 + 0.01 * i); pts.emplace_back(20 + 0.02 * i, -3 + 0.01 * i); }
  const auto cl = cluster_bev(pts);
  ASSERT_EQ(cl.size(), 2u);
  CentroidTracker tr;
  const auto r1 = tr.update(cl), r2 = tr.update(cl);
  ASSERT_EQ(r1.size(), r2.size());
  for (size_t i = 0; i < r1.size(); ++i) EXPECT_EQ(r1[i].first, r2[i].first);
}

TEST(Preprocess, ResizeAndChannelOrder) {
  std::vector<uint8_t> img(size_t(W) * H * 4);
  for (size_t i = 0; i < img.size(); i += 4) { img[i] = 10; img[i + 1] = 20; img[i + 2] = 30; img[i + 3] = 255; }   // B G R A
  std::vector<uint8_t> out(size_t(3) * MODEL_W * MODEL_H);
  preprocess_rgb_chw(img.data(), W, H, W * 4, 4, true, false, 0, 0, out.data());
  EXPECT_EQ(out[0], 30); EXPECT_EQ(out[size_t(MODEL_W) * MODEL_H], 20); EXPECT_EQ(out[2 * size_t(MODEL_W) * MODEL_H], 10);
  preprocess_rgb_chw(img.data(), W, H, W * 4, 4, true, true, 40, 40, out.data());   // crop mode
  EXPECT_EQ(out[0], 30);
}
