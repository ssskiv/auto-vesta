// GoogleTest port of test/test_logic.py.

#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "offroad_nav2/dbw_adapter.hpp"
#include "offroad_nav2/vehicle_logic.hpp"

namespace {
constexpr double kL = 2.648;
}  // namespace

TEST(VehicleLogic, NegativeSpeedClampedToZero) {
  offroad::ZeroClampWatchdog w;
  for (double v : {-5.0, -0.001, -1e-9}) {
    EXPECT_EQ(w.update(v, 0.3, 1.0).v, 0.0);
    EXPECT_EQ(w.output(1.0).v, 0.0);
  }
  EXPECT_EQ(w.update(2.0, 0.3, 1.0).v, 2.0);
  EXPECT_EQ(w.clamped(), 3u);
}

TEST(VehicleLogic, SteerExactAtan) {
  const double cases[][2] = {{2.0, 0.3}, {5.0, -0.2}, {1.0, 0.1}};
  for (const auto& c : cases) {
    const double v = c[0];
    const double om = c[1];
    EXPECT_EQ(offroad::steer_from_twist(v, om, kL, 0.0, 10.0), std::atan(kL * om / v));
  }
}

TEST(VehicleLogic, SteerClampHoldAndRateLimit) {
  EXPECT_EQ(offroad::steer_from_twist(1.0, 5.0, kL, 0.0, 0.6), 0.6);
  EXPECT_EQ(offroad::steer_from_twist(1.0, -5.0, kL, 0.0, 0.6), -0.6);
  // standstill: hold previous steer
  EXPECT_EQ(offroad::steer_from_twist(0.05, 5.0, kL, 0.2, 0.6), 0.2);
  // rate limit: max_rate * dt = 0.05 rad step; v_min kept at its default 0.1
  EXPECT_LT(std::fabs(offroad::steer_from_twist(1.0, 5.0, kL, 0.0, 0.6, 0.1, 0.5, 0.1) - 0.05), 1e-12);
}

TEST(VehicleLogic, Watchdog300ms) {
  offroad::ZeroClampWatchdog w(0.3);
  EXPECT_TRUE(w.output(0.0).stale);  // nothing received yet: stale -> zero
  w.update(3.0, 0.1, 10.0);

  auto o = w.output(10.29);
  EXPECT_FALSE(o.stale);
  EXPECT_EQ(o.v, 3.0);
  EXPECT_EQ(o.w, 0.1);

  o = w.output(10.31);
  EXPECT_TRUE(o.stale);
  EXPECT_EQ(o.v, 0.0);
  EXPECT_EQ(o.w, 0.0);

  w.update(3.0, 0.1, 10.4);
  EXPECT_FALSE(w.output(10.5).stale);
}

TEST(VehicleLogic, WheelSpeed) {
  offroad::WheelSpeed ws(0.315);
  EXPECT_FALSE(ws.update({0.0, 0.0}, 0.0).has_value());
  const auto v = ws.update({1.0, 1.0}, 0.1);  // 10 rad/s * 0.315
  ASSERT_TRUE(v.has_value());
  EXPECT_NEAR(*v, 3.15, 1e-9);
}

TEST(DbwStub, EngageAndEstop) {
  std::vector<std::string> logs;
  offroad::LoggingDbwAdapter a([&logs](offroad::LogLevel, const std::string& msg) { logs.push_back(msg); });

  a.send(0.1, 2.0, 0.0);
  auto st = a.status();
  ASSERT_TRUE(st.last.has_value());
  EXPECT_DOUBLE_EQ(st.last->steer_rad, 0.1);
  EXPECT_DOUBLE_EQ(st.last->speed_mps, 2.0);
  EXPECT_DOUBLE_EQ(st.last->accel, 0.0);
  EXPECT_FALSE(st.estop);
  EXPECT_FALSE(st.engaged);

  a.emergency_stop();
  a.send(0.1, 2.0, 0.0);  // zeroed while e-stopped
  st = a.status();
  EXPECT_TRUE(st.estop);
  ASSERT_TRUE(st.last.has_value());
  EXPECT_DOUBLE_EQ(st.last->steer_rad, 0.0);
  EXPECT_DOUBLE_EQ(st.last->speed_mps, 0.0);
  EXPECT_DOUBLE_EQ(st.last->accel, 0.0);

  ASSERT_FALSE(logs.empty());
  EXPECT_EQ(logs.back().rfind("DBW stub send", 0), 0u);
  bool saw_estop = false;
  for (const auto& l : logs) {
    saw_estop = saw_estop || l == "DBW stub EMERGENCY STOP";
  }
  EXPECT_TRUE(saw_estop);
}
