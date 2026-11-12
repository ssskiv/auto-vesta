#pragma once
// Pure C++ port of offroad_vehicle_interface/logic.py (no ROS includes).

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace offroad {

struct Cmd {
  double v;
  double w;
};

struct WatchdogOutput {
  double v;
  double w;
  bool stale;
};

// linear.x = max(linear.x, 0); if no command arrived for `timeout` s the output is a zero command.
class ZeroClampWatchdog {
 public:
  explicit ZeroClampWatchdog(double timeout = 0.3);

  // Stores the (clamped) command and returns it.
  Cmd update(double v, double w, double now);

  // Stale (no command yet, or older than timeout) -> zero command with stale = true.
  WatchdogOutput output(double now) const;

  std::size_t clamped() const { return clamped_; }
  double timeout() const { return timeout_; }

 private:
  double timeout_;
  std::optional<double> last_in_;
  double v_ = 0.0;
  double w_ = 0.0;
  std::size_t clamped_ = 0;
};

// delta = atan(L*omega/v) for v > v_min, else hold prev_steer. Clamp to +-max_steer, optional rate limit
// (applied only when both max_rate and dt are given).
double steer_from_twist(double v, double omega, double wheelbase, double prev_steer, double max_steer,
                        double v_min = 0.1, std::optional<double> max_rate = std::nullopt,
                        std::optional<double> dt = std::nullopt);

// Forward speed from cumulative wheel angles (rad): v = sign * r * mean(d theta) / dt.
class WheelSpeed {
 public:
  explicit WheelSpeed(double radius, double sign = 1.0);

  // Returns nullopt on the first sample, on a non-increasing timestamp, or when no angle pairs are available.
  std::optional<double> update(const std::vector<double>& angles, double t);

 private:
  struct Prev {
    std::vector<double> angles;
    double t;
  };
  double r_;
  double sign_;
  std::optional<Prev> prev_;
};

}  // namespace offroad
