#include <gtest/gtest.h>

#include <random>

#include "offroad_nav2/geometry.hpp"

using namespace offroad;

static Mat3 K0() { Mat3 K = Mat3::Identity(); K(0, 0) = K(1, 1) = 640; K(0, 2) = 320; K(1, 2) = 180; return K; }

TEST(Geometry, LiftDepthWithin1mm) {
  const Mat3 K = K0();
  for (auto [u, v, z] : {std::tuple{320., 180., 5.0}, {100., 50., 12.3}, {600., 300., 3.1}}) {
    const Vec3 p = backproject(u, v, z, K);
    double u2, v2;
    project(p, K, u2, v2);
    EXPECT_NEAR(u2, u, 1e-9); EXPECT_NEAR(v2, v, 1e-9); EXPECT_NEAR(p.z(), z, 1e-3);
    EXPECT_NEAR(p.x(), (u - 320) * z / 640, 1e-3);
  }
}

TEST(Geometry, ScaleIntrinsicsAndCoords) {
  Mat3 K = Mat3::Identity(); K(0, 0) = 1000; K(1, 1) = 900; K(0, 2) = 640; K(1, 2) = 360;
  const Mat3 Ks = scale_intrinsics(K, 1280, 720);
  EXPECT_NEAR(Ks(0, 0), 500, 1e-9); EXPECT_NEAR(Ks(1, 1), 400, 1e-9); EXPECT_NEAR(Ks(0, 2), 320, 1e-9); EXPECT_NEAR(Ks(1, 2), 160, 1e-9);
  const Vec3 p(2.0, -0.5, 10.0);
  double u, v, um, vm, ub, vb;
  project(p, K, u, v); project(p, Ks, um, vm);
  model_to_image_coords(um, vm, 1280, 720, false, 0, 0, ub, vb);
  EXPECT_NEAR(ub, u, 1e-9); EXPECT_NEAR(vb, v, 1e-9);
  const Mat3 Kc = scale_intrinsics(K, 1280, 720, true, 40, 40);
  EXPECT_NEAR(Kc(1, 1), 450, 1e-9); EXPECT_NEAR(Kc(1, 2), (360 - 40) * 0.5, 1e-9);
}

static Mat4 cam() { return body_to_optical_T(Vec3(1.6, 0, 1.3), 5.0 * M_PI / 180); }

TEST(Geometry, GroundPointRoundTripUnder1cm) {
  const Mat3 K = K0(); const Mat4 T = cam();
  for (Vec3 g : {Vec3(6, .5, 0), Vec3(10, -2, 0), Vec3(4, 1.2, 0), Vec3(20, 3, 0)}) {
    const Vec3 pc = (T.inverse() * g.homogeneous()).head<3>();
    double u, v;
    project(pc, K, u, v);
    const auto back = pixel_to_ground(u, v, K, T);
    ASSERT_TRUE(back.has_value());
    EXPECT_LT((*back - g).norm(), 0.01);
    EXPECT_LT((transform(T, backproject(u, v, pc.z(), K)) - g).norm(), 0.01);
  }
}

TEST(Geometry, LeftOfCentreIsPositiveYAndOpticalAxis) {
  const Mat3 K = K0(); const Mat4 T = cam();
  EXPECT_GT(transform(T, backproject(100, 180, 8.0, K)).y(), 0);
  EXPECT_LT(transform(T, backproject(500, 180, 8.0, K)).y(), 0);
  const Vec3 z = T.block<3, 1>(0, 2);
  EXPECT_GT(z.x(), 0.99); EXPECT_LT(z.z(), 0);
  EXPECT_NEAR(std::asin(-z.z()) * 180 / M_PI, 5.0, 1e-6);
}

TEST(Geometry, BottomCentreRayAndHorizon) {
  const Mat3 K = K0(); const Mat4 T = cam();
  const auto g = pixel_to_ground(320, 319, K, T);
  ASSERT_TRUE(g.has_value());
  const double ang = std::atan2(139.0, 640.0) + 5.0 * M_PI / 180;
  EXPECT_NEAR(g->x(), 1.6 + 1.3 / std::tan(ang), 0.01);
  EXPECT_NEAR(g->y(), 0, 1e-6);
  const auto h = horizon_row(K, T);
  ASSERT_TRUE(h.has_value());
  EXPECT_NEAR(*h, 180 - 640 * std::tan(5.0 * M_PI / 180), 1e-6);
  EXPECT_TRUE(above_horizon(*h - 5, K, T)); EXPECT_FALSE(above_horizon(*h + 20, K, T));
  EXPECT_FALSE(pixel_to_ground(320, *h - 5, K, T).has_value());
}

TEST(Geometry, RansacRecoversPlane) {
  std::mt19937 rng(1);
  std::normal_distribution<double> noise(0, 0.01);
  std::uniform_real_distribution<double> uni(-10, 10), uo(-3, 3);
  Vec3 nt(std::sin(4 * M_PI / 180), 0, std::cos(4 * M_PI / 180));
  Points pts;
  for (int i = 0; i < 2000; ++i) { const double x = uni(rng), y = uni(rng); pts.emplace_back(x, y, -(nt.x() * x + nt.y() * y) / nt.z() + noise(rng)); }
  for (int i = 0; i < 400; ++i) pts.emplace_back(uo(rng), uo(rng), uo(rng) + 1.0);
  const auto p = fit_ground_plane(pts, Vec3::UnitZ());
  ASSERT_TRUE(p.has_value());
  EXPECT_LT(std::acos(std::clamp(p->n.dot(nt), -1.0, 1.0)) * 180 / M_PI, 1.0);
  int in = 0; for (int i = 0; i < 2000; ++i) in += p->inliers[i];
  EXPECT_GT(in / 2000.0, 0.9);
}

TEST(Geometry, ContourFlags) {
  uint8_t bins[CONTOUR_RAYS]; std::fill(bins, bins + CONTOUR_RAYS, 10);
  bins[5] = 45; bins[0] = 40; bins[18] = 20;
  const auto e = contour_endpoints(bins);
  EXPECT_TRUE(e.clear_only[5]); EXPECT_TRUE(e.clear_only[0]); EXPECT_FALSE(e.clear_only[18]);
  EXPECT_NEAR(e.row[18], 119, 1e-6); EXPECT_NEAR(e.col[18], 320, 1e-6);
  for (int i = 0; i < CONTOUR_RAYS; ++i) { EXPECT_GE(e.row[i], 0); EXPECT_LE(e.row[i], 319); EXPECT_GE(e.col[i], 0); EXPECT_LE(e.col[i], 639); }
}

TEST(Geometry, MedianDepthInvalidAndWindow) {
  std::vector<float> d(20 * 20, 5.0f);
  d[10 * 20 + 10] = std::nanf(""); d[9 * 20 + 9] = 0.f; d[11 * 20 + 11] = 99.f;
  DepthView v{d.data(), 20, 20, 20};
  EXPECT_EQ(median_depth(v, 10, 10), 5.0); EXPECT_EQ(median_depth(v, 0, 0), 5.0); EXPECT_TRUE(std::isnan(median_depth(v, 50, 50)));
  std::fill(d.begin(), d.end(), std::nanf(""));
  EXPECT_TRUE(std::isnan(median_depth(v, 5, 5)));
}
