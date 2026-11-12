#include "offroad_nav2/vehicle_logic.hpp"

#include <algorithm>
#include <cmath>

namespace offroad {

ZeroClampWatchdog::ZeroClampWatchdog(double timeout) : timeout_(timeout) {}

Cmd ZeroClampWatchdog::update(double v, double w, double now) {
  last_in_ = now;
  if (v < 0.0) {
    v = 0.0;
    ++clamped_;
  }
  v_ = v;
  w_ = w;
  return {v_, w_};
}

WatchdogOutput ZeroClampWatchdog::output(double now) const {
  if (!last_in_ || now - *last_in_ > timeout_) {
    return {0.0, 0.0, true};
  }
  return {v_, w_, false};
}

double steer_from_twist(double v, double omega, double wheelbase, double prev_steer, double max_steer,
                        double v_min, std::optional<double> max_rate, std::optional<double> dt) {
  double delta = (v > v_min) ? std::atan(wheelbase * omega / v) : prev_steer;
  delta = std::max(-max_steer, std::min(max_steer, delta));
  if (max_rate && dt) {
    const double step = *max_rate * *dt;
    delta = std::max(prev_steer - step, std::min(prev_steer + step, delta));
  }
  return delta;
}

WheelSpeed::WheelSpeed(double radius, double sign) : r_(radius), sign_(sign) {}

std::optional<double> WheelSpeed::update(const std::vector<double>& angles, double t) {
  std::optional<double> v;
  if (prev_ && t > prev_->t) {
    // Python zip() semantics: only the common prefix of the two angle vectors is used.
    const std::size_t n = std::min(angles.size(), prev_->angles.size());
    if (n > 0) {
      double sum = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        sum += angles[i] - prev_->angles[i];
      }
      v = sign_ * r_ * (sum / static_cast<double>(n)) / (t - prev_->t);
    }
  }
  prev_ = Prev{angles, t};
  return v;
}

}  // namespace offroad
