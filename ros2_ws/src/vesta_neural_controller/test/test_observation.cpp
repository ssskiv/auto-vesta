// Checks observation.cpp against vesta_rl/obs.py on fixtures written by
// `ros2 run vesta_rl make_obs_fixture`.
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <vector>

#include "vesta_neural_controller/observation.hpp"

using vesta_neural_controller::CostmapView;
using vesta_neural_controller::PathTracker;
namespace obs = vesta_neural_controller::obs;

TEST(Observation, MatchesPythonReference)
{
  std::ifstream in(FIXTURE);
  ASSERT_TRUE(in.good()) << "missing " << FIXTURE;
  int n_cases;
  in >> n_cases;
  ASSERT_GT(n_cases, 0);

  for (int c = 0; c < n_cases; ++c) {
    SCOPED_TRACE("case " + std::to_string(c));
    int sx, sy;
    double ox, oy, res;
    in >> sx >> sy >> ox >> oy >> res;
    std::vector<unsigned char> data(sx * sy);
    for (auto & d : data) {
      int v;
      in >> v;
      d = static_cast<unsigned char>(v);
    }
    std::size_t n_path;
    in >> n_path;
    std::vector<double> path(2 * n_path);
    for (auto & p : path) {
      in >> p;
    }
    std::size_t idx_in, idx_out;
    double px, py, yaw, v, w, last_v, last_delta, max_speed;
    in >> idx_in >> px >> py >> yaw >> v >> w >> last_v >> last_delta >> max_speed >> idx_out;
    std::vector<float> want_patch(obs::PATCH_SIZE * obs::PATCH_SIZE), want_vec(obs::VEC_SIZE);
    for (auto & p : want_patch) {
      in >> p;
    }
    for (auto & p : want_vec) {
      in >> p;
    }
    double want_limit;
    in >> want_limit;
    ASSERT_TRUE(in.good());

    const CostmapView view{data.data(), sx, sy, ox, oy, res};
    std::vector<float> patch(want_patch.size());
    vesta_neural_controller::buildPatch(view, px, py, yaw, patch.data());
    int mismatched = 0;
    for (std::size_t i = 0; i < patch.size(); ++i) {
      mismatched += patch[i] != want_patch[i];
    }
    EXPECT_EQ(mismatched, 0);

    PathTracker tracker(path);
    tracker.idx = idx_in;
    std::vector<float> vec(obs::VEC_SIZE);
    const auto info = vesta_neural_controller::buildVec(
      tracker, px, py, yaw, v, w, last_v, last_delta, max_speed, vec.data());
    EXPECT_EQ(tracker.idx, idx_out);
    EXPECT_NEAR(vesta_neural_controller::speedLimit(tracker, info.s, max_speed), want_limit, 1e-9);
    for (int i = 0; i < obs::VEC_SIZE; ++i) {
      EXPECT_NEAR(vec[i], want_vec[i], 1e-6) << "vec[" << i << "]";
    }
  }
}

TEST(Observation, ActionMappingAndRateLimit)
{
  double v, delta;
  vesta_neural_controller::actionToCommand(1.0f, -1.0f, 5.0, v, delta);
  EXPECT_DOUBLE_EQ(v, 5.0);
  EXPECT_DOUBLE_EQ(delta, -obs::MAX_STEER);
  vesta_neural_controller::actionToCommand(-2.0f, 0.5f, 5.0, v, delta);
  EXPECT_DOUBLE_EQ(v, 0.0);
  EXPECT_DOUBLE_EQ(delta, 0.3);
  vesta_neural_controller::rateLimit(v, delta, 1.0, 0.0);
  EXPECT_DOUBLE_EQ(v, 1.0 - obs::MAX_DV);
  EXPECT_DOUBLE_EQ(delta, obs::MAX_DDELTA);
}

TEST(Observation, WrapAngleMatchesPython)
{
  EXPECT_NEAR(vesta_neural_controller::wrapAngle(3 * M_PI / 2), -M_PI / 2, 1e-12);
  EXPECT_NEAR(vesta_neural_controller::wrapAngle(-3 * M_PI / 2), M_PI / 2, 1e-12);
  EXPECT_NEAR(vesta_neural_controller::wrapAngle(0.3), 0.3, 1e-12);
}
